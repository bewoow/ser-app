-- This query extracts the duration of mechanical ventilation
-- The main goal of the query is to aggregate sequential ventilator settings
-- into single mechanical ventilation "events". The start and end time of these
-- events can then be used for various purposes: calculating the total duration
-- of mechanical ventilation, cross-checking values (e.g. PaO2:FiO2 on vent), etc

-- The query's logic is roughly:
--    1) The presence of a mechanical ventilation setting starts a new ventilation event
--    2) Any instance of a setting in the next 8 hours continues the event
--    3) Certain elements end the current ventilation event
--        a) documented extubation ends the current ventilation
--        b) initiation of non-invasive vent and/or oxygen ends the current vent
-- The ventilation events are numbered consecutively by the `num` column.


-- First, create a temporary table to store relevant data from CHARTEVENTS.

CREATE TABLE DATABASE.vent_durations AS
WITH vd0 AS
(
    SELECT
    icustay_id
    -- this carries over the previous charttime which had a mechanical ventilation event
    , CASE
        WHEN MechVent=1 THEN
            LAG(CHARTTIME, 1) OVER
            (PARTITION BY icustay_id, MechVent ORDER BY charttime)
        ELSE NULL
        END AS charttime_lag
    , charttime
    , MechVent
    , OxygenTherapy
    , Extubated
    , SelfExtubated
    FROM DATABASE.ventsettings
)
, vd1 AS
(
    SELECT
    icustay_id
    , charttime_lag
    , charttime
    , MechVent
    , OxygenTherapy
    , Extubated
    , SelfExtubated
    -- if this is a mechanical ventilation event, we calculate the time since the last event
    , CASE
        -- if the current observation indicates mechanical ventilation is present
        -- calculate the time since the last vent event
        WHEN MechVent=1 THEN
            date_diff('hour', charttime_lag, CHARTTIME)
        ELSE NULL
    END AS ventduration
    -- now we determine if the current mech vent event is a "new", i.e. they've just been intubated
    , CASE
        -- if there is an extubation flag, we mark any subsequent ventilation as a new ventilation event
        --when Extubated = 1 then 0 -- extubation is *not* a new ventilation event, the *subsequent* row is
        WHEN
            LAG(Extubated,1)
                OVER
                (
                    PARTITION BY icustay_id, CASE WHEN MechVent=1 OR Extubated=1 THEN 1 ELSE 0 END
                    ORDER BY charttime
                )
            = 1 THEN 1
        -- if patient has initiated oxygen therapy, and is not currently vented, start a newvent
        WHEN MechVent = 0 AND OxygenTherapy = 1 THEN 1
        -- if there is less than 8 hours between vent settings, we do not treat this as a new ventilation event
        WHEN date_diff('second', charttime_lag, CHARTTIME)/60.0/60.0 > 8.0 THEN 1
        ELSE 0
    END AS newvent
    -- use the staging table with only vent settings from chart events
    FROM vd0 ventsettings
)
, vd2 AS
(
    SELECT vd1.*
    -- create a cumulative sum of the instances of new ventilation
    -- this results in a monotonic integer assigned to each instance of ventilation
    , CASE WHEN MechVent=1 OR Extubated = 1 THEN
        SUM( newvent )
        OVER ( PARTITION BY icustay_id ORDER BY charttime )
    ELSE NULL END
    AS ventnum -- a different ventnum number indicates a new vent event! (used in the group by below)
    --- now we convert CHARTTIME of ventilator settings into durations
    FROM vd1
)
-- create the durations for each mechanical ventilation instance
SELECT icustay_id
  -- regenerate ventnum so it's sequential
    , ROW_NUMBER() over (PARTITION BY icustay_id ORDER BY ventnum) AS ventnum
    , min(charttime) AS starttime
    , max(charttime) AS endtime
    , date_diff('second', min(charttime), max(charttime))/60.0000000/60.0000000 AS duration_hours
FROM vd2
GROUP BY icustay_id, ventnum
HAVING min(charttime) != max(charttime)
-- patient had to be mechanically ventilated at least once
-- i.e. max(mechvent) should be 1
-- this excludes a frequent situation of NIV/oxygen before intub
-- in these cases, ventnum=0 and max(mechvent)=0, so they are ignored
AND max(mechvent) = 1
ORDER BY icustay_id, ventnum;
