# Usage:
#   praat --run scripts/praat_syllable_nuclei.praat input.wav output.json
#
# Emits a compact JSON document:
#   {"silences":[{"t0":0.42,"t1":0.74}],"nuclei":[{"t":1.23,"intensity_db":68.1}]}

form Extract silence and vowel nuclei
    sentence wav_path
    sentence output_json
    real silence_threshold_db -25
    positive min_silence_seconds 0.25
    positive min_nucleus_distance_seconds 0.12
endform

Read from file: wav_path$
sound = selected("Sound")
duration = Get total duration

To Intensity: 75, 0, "yes"
intensity = selected("Intensity")

selectObject: sound
To TextGrid (silences): silence_threshold_db, min_silence_seconds, 0.08, "silent", "sounding"
textgrid = selected("TextGrid")

json$ = "{""silences"":["
interval_count = Get number of intervals: 1
first_silence = 1
for i from 1 to interval_count
    label$ = Get label of interval: 1, i
    if label$ = "silent"
        t0 = Get start time of interval: 1, i
        t1 = Get end time of interval: 1, i
        if t1 - t0 >= min_silence_seconds
            if first_silence = 0
                json$ = json$ + ","
            endif
            json$ = json$ + "{""t0"":" + fixed$(t0, 4) + ",""t1"":" + fixed$(t1, 4) + "}"
            first_silence = 0
        endif
    endif
endfor

json$ = json$ + "],""nuclei"":["
selectObject: intensity
time = 0
last_nucleus = -999
first_nucleus = 1
while time < duration
    value = Get value at time: time, "Cubic"
    left_time = max(0, time - 0.03)
    right_time = min(duration, time + 0.03)
    left = Get value at time: left_time, "Cubic"
    right = Get value at time: right_time, "Cubic"
    if value <> undefined and left <> undefined and right <> undefined
        if value > left and value >= right and value > silence_threshold_db and time - last_nucleus >= min_nucleus_distance_seconds
            if first_nucleus = 0
                json$ = json$ + ","
            endif
            json$ = json$ + "{""t"":" + fixed$(time, 4) + ",""intensity_db"":" + fixed$(value, 2) + "}"
            last_nucleus = time
            first_nucleus = 0
        endif
    endif
    time = time + 0.01
endwhile
json$ = json$ + "]}"

filedelete 'output_json$'
fileappend 'output_json$' 'json$'

selectObject: textgrid
Remove
selectObject: intensity
Remove
selectObject: sound
Remove
