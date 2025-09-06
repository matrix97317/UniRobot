set -e

# This environment variable is a temporary hack on cu114 for cpu affinity.
export KMP_AFFINITY=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export UNIROBOT_launch_mode=spawn

DEVICE_NUM=$1
if [ "$DEVICE_NUM" = "" ]; then
  echo "You must set device num"
  exit -1
else
  echo "You set device num $DEVICE_NUM"
fi

echo pipenv run unirobot brain-run ${@:1}
pipenv run unirobot brain-run ${@:1}
