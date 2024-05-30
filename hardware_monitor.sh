mpstat -P ALL 1 2>&1 > $1 &
pid=$!
shift
time $*
kill $pid


