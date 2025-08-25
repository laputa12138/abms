nohup python main.py --web --host 0.0.0.0 --port 1877 > app/log.txt 2>&1 &
tail -f app/log.txt