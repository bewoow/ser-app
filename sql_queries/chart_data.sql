SELECT ie.subject_id, ie.icustay_id, ce.charttime
, CASE
    WHEN itemid IN (211,220045) AND valuenum > 0 AND valuenum < 300 THEN 'HR' -- Heart Rate
    WHEN itemid IN (51,442,455,6701,220179,220050,3313,3315,3321) AND valuenum > 0 AND valuenum < 400 THEN 'SBP'-- Sys BP
    WHEN itemid IN (8368,8440,8441,8555,220180,220051,8502,8503,8506) AND valuenum > 0 AND valuenum < 300 THEN 'DBP' -- DiasBP
    WHEN itemid IN (456,52,6702,443,220052,220181,225312) AND valuenum > 0 AND valuenum < 300 THEN 'MBP' -- MeanBP
    WHEN itemid IN (615,618,220210,224690,3603) AND valuenum > 0 AND valuenum < 70 THEN 'RR' -- RespRate
    WHEN itemid IN (223761,678) AND valuenum > 70 AND valuenum < 120  THEN 'Temp' -- Temp in F, converted to degC IN valuenum call
    WHEN itemid IN (223762,676) AND valuenum > 10 AND valuenum < 50  THEN 'Temp' -- Temp in C
    WHEN itemid IN (646,220277) AND valuenum > 25 AND valuenum <= 100 THEN 'SpO2' -- SpO2
    WHEN itemid IN (682,224685) AND valuenum > 0 AND valuenum <= 8000 THEN 'VTobs' -- Tidal volume observed
    WHEN itemid IN (684,3688,224686,224421) AND valuenum > 0 and valuenum <= 8000 THEN 'VTspot' -- Tidal volume spot
    WHEN itemid IN (535, 224695) AND valuenum > 0 AND valuenum <= 80 THEN 'PIP' -- Peak inspiratory pressure
    WHEN itemid IN (506, 220339) AND valuenum > 0 AND valuenum <= 40 THEN 'PEEP' -- PEEP
    WHEN itemid = 198 AND valuenum >= 3 AND valuenum <= 15 THEN 'GCStot' -- Glasgow Coma Scale
    WHEN itemid = 220739 AND valuenum >= 1 AND valuenum <= 4 THEN 'GCSeye' -- GCS Eye opening
    WHEN itemid = 223900 AND valuenum >= 1 AND valuenum <= 5 THEN 'GCSverbal' -- GCS - Verbal Response
    WHEN itemid = 223901 AND valuenum >= 1 AND valuenum <= 6 THEN 'GCSmotor' -- GCS - Motor Response
    WHEN itemid = 778 AND valuenum >= 1 AND valuenum <= 60 THEN 'PaCO2' -- PaCO2
    WHEN itemid = 779 AND valuenum >= 1 AND valuenum <= 150 THEN 'PaO2' -- PaO2
    WHEN itemid IN (190,727) AND valuenum >= 0.21 AND valuenum <= 1 THEN 'FiO2' -- FiO2
    WHEN itemid IN (113,220074) AND valuenum >= 0 AND valuenum <= 30 THEN 'CVP' -- Central Venous Pressure
    WHEN itemid = 491 AND valuenum >=0 AND valuenum <= 50 THEN 'PAP' -- Pulmonary Arterial Pressure (mean)
    WHEN itemid IN (492,220059) AND valuenum >=0 AND valuenum <= 50 THEN 'PAPSys' -- Pulmonary Arterial Pressure (systolic)
    ELSE NULL END
  AS vital_sign
, CASE
    -- convert F to C
    WHEN itemid IN (223761,678) THEN (valuenum - 32) / 1.8
    -- convert tidal volume from liters to ml
    WHEN itemid = 3688 THEN valuenum * 1000
    -- convert FiO2 to percentage
    WHEN itemid IN (190,727) THEN valuenum * 100
    ELSE valuenum END
  AS valuenum
, CASE
    WHEN itemid IN (211,220045) THEN 'bpm' -- heart rate (bpm)
    WHEN itemid IN (51,442,455,6701,220179,220050,3313,3315,3321) THEN 'mmHg' -- SBP
    WHEN itemid IN (8368,8440,8441,8555,220180,220051,8502,8503,8506) THEN 'mmHg' -- DBP
    WHEN itemid IN (456,52,6702,443,220052,220181,225312) THEN 'mmHg' -- MBP
    WHEN itemid IN (615,618,220210,224690,3603) THEN 'bpm' -- RR (bpm)
    WHEN itemid IN (223761,678,223762,676) THEN 'C' -- Temp
    WHEN itemid IN (646,220277,190,727) THEN '%' -- SpO2, FiO2
    WHEN itemid IN (682,224685,684,3688,224686,224421) THEN 'ml' -- tidal volume
    WHEN itemid IN (535, 224695,506, 220339) THEN 'cmH2O' -- PIP, PEEP
    WHEN itemid IN (778,779) THEN 'mmHg' -- PaO2, PaCO2
    WHEN itemid IN (113,220074,491,492,220059) THEN 'mmHg' -- CVP, PAP, PAPSys
    ELSE NULL END
  AS unit

FROM DATABASE.icustays AS ie
LEFT JOIN DATABASE.chartevents AS ce
ON ie.subject_id = ce.subject_id AND ie.hadm_id = ce.hadm_id AND ie.icustay_id = ce.icustay_id
AND ce.charttime BETWEEN timestamp 'STARTTIME' AND timestamp 'ENDTIME'
-- exclude rows marked AS error
AND ce.error IS DISTINCT FROM 1
WHERE ce.itemid IN
(
    -- HEART RATE
    211, -- "Heart Rate"
    220045, -- "Heart Rate"

    -- Systolic/diastolic Blood pressure

    51, -- Arterial BP [Systolic]
    442, -- Manual BP [Systolic]
    455, -- NBP [Systolic]
    6701, -- Arterial BP #2 [Systolic]
    220179, -- Non Invasive Blood Pressure systolic
    220050, -- Arterial Blood Pressure systolic
    3313, -- BP Cuff [Systolic]
    3315, -- BP Left Arm [Systolic]
    3321, -- BP Right Arm [Systolic]

    8368, -- Arterial BP [Diastolic]
    8440, -- Manual BP [Diastolic]
    8441, -- NBP [Diastolic]
    8555, -- Arterial BP #2 [Diastolic]
    220180, -- Non Invasive Blood Pressure diastolic
    220051, -- Arterial Blood Pressure diastolic
    8502, -- BP Cuff [Diastolic]
    8503, -- BP Left Arm [Diastolic]
    8506, -- BP Right Arm [Diastolic]

    -- MEAN ARTERIAL PRESSURE
    456, -- "NBP Mean"
    52, -- "Arterial BP Mean"
    6702, -- Arterial BP Mean #2
    443, -- Manual BP Mean(calc)
    220052, -- "Arterial Blood Pressure mean"
    220181, -- "Non Invasive Blood Pressure mean"
    225312, -- "ART BP mean"

    -- RESPIRATORY RATE
    618,-- Respiratory Rate
    615,-- Resp Rate (Total)
    220210,-- Respiratory Rate
    224690, -- Respiratory Rate (Total)
    3603, -- Resp Rate

    -- SPO2, peripheral
    646, 220277,

    -- TIDAL VOLUME (observed)
    682, -- Tidal Volume (Obser)
    224685, -- Tidal Volume (observed)

    -- TIDAL VOLUME (spontaneous)
    684, -- Tidal Volume (Spont)
    3688, -- Vt [Spontaneous]
    224686, -- Tidal Volume (spontaneous)
    224421, -- Spont Vt

    -- Peak inspiratory pressure
    535, 224695,

    -- PEEP
    506, 220339,

    -- Glasgow Coma Scale
    198, -- GCS total
    220739, -- GCS - Eye response
    223900,-- GCS - Verbal Response
    223901, -- GCS - Motor Response

    -- TEMPERATURE
    223762, -- "Temperature Celsius"
    676,	-- "Temperature C"
    223761, -- "Temperature Fahrenheit"
    678, --	"Temperature F"

    778, -- PaCO2
    779, -- PaO2
    190,727, -- FiO2
    113,220074, -- CVP
    491, -- PAP mean
    492,220059 -- PAPSys
) AND ie.icustay_id = ICUSTAY_ID
