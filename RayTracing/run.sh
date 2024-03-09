# !/bin/bash

# rm -rf bin png vid1.mp4
# mkdir bin png

# echo "Running solution"
time ./"$1" $2 < ./in/normal.in
# echo "Running toPng.cpp"
pids=() # Массив для хранения PID запущенных процессов

for ((k = 0; k < 100000; k++ )); do
    bin="./bin/$k.bin"
    if test -f "$bin"; then
        i=$(printf "%03d" $((k+1)))
        png="./png/$i.png"
        ./"$3" "$bin" "$png" &
        pids+=($!) # Сохранение PID процесса в массив

        if (( k % 10 == 0 )); then
            for pid in "${pids[@]}"; do
                wait $pid
            done
            pids=() 
        fi
    else
        break
    fi
done

for pid in "${pids[@]}"; do
    wait $pid
done

cd png
ffmpeg -framerate 60 -i %03d.png -s 1080x720 -pix_fmt yuv420p ../vid1.mp4

# ffmpeg -framerate 60 -i %03d.png -s 1920x1080 -pix_fmt yuv420p ../vid1.mp4
