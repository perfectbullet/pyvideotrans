ffmpeg -y -i D:/zjpython_work/pyvideotrans-main/ffmpeg/to_compose/novoice.mp4 -i D:/zjpython_work/pyvideotrans-main/ffmpeg/to_compose/zhvoice.wav -filter_complex [0:a][1:a]amerge=inputs=2[aout] -map 0:v -map [aout] -c:v libx264 -c:a aac D:/zjpython_work/pyvideotrans-main/tmp/1702351030.4315577.mp4

ffmpeg -hide_banner -y -i D:/zjpython_work/pyvideotrans-main/tmp/1702351030.4315577.mp4 -vf subtitles=D\\:/zjpython_work/pyvideotrans-main/ffmpeg/to_compose/zh-cn.srt C:/Users/gx/Videos/pyvideotrans/hebing-novoice.mp4/novoice.mp4.mp4


ffmpeg -hide_banner -y -i 1702351030.4315577.mp4 -vf subtitles=en.srt out.mp4




