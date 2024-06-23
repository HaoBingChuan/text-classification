#!/bin/sh
export PATH="/opt/venv/bin:$PATH"
if [ ! "$LOG_LEVEL" ]; then
  LOG_LEVEL=INFO
fi
if [ ! "$LOG_FILE" ]; then
  LOG_FILE=work.log
fi

# if [ "$TOOLS" ]; then
#   python main.py
if [ "$SERVER" ]; then
  python server.py
elif [ "$TASK" ]; then
	celery -A "$TASK" worker -c 10 $OPTIONS -l "$LOG_LEVEL" -f "$LOG_FILE"
fi


