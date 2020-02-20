#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

print_green() {
    [ -t 1 ] && echo -e "\e[32m$@\e[39m" || echo "$@"
}

print_red() {
    [ -t 1 ] && echo -e "\e[31m$@\e[39m" || echo "$@"
}

TECKYL="./bin/teckyl"

[ -x "$TECKYL" ] || \
    die "Could not find teckyl binary." \
	"Please run this script from the build directory."

BASE_DIR="$(dirname "${BASH_SOURCE[0]}")"

TMP_LOGFILE="/tmp/teckyl-test.$$.log"
trap "{ rm -f \"$TMP_LOGFILE\" ; }" EXIT

for MODE in good bad
do
    TEST_DIR="$BASE_DIR/tests/inputs/$MODE"

    for EXTRA_ARGS in "" "-force-std-loops"
    do
	find "$TEST_DIR" -type f -name "*.tc" -print0 | sort | \
	    while IFS= read -r -d '' SRC_FILE
	    do
		printf '%s' "Running teckyl $EXTRA_ARGS on $SRC_FILE... "

		# Suppress error messages from the shell
		exec 2> /dev/null
		"$TECKYL" -emit=mlir $EXTRA_ARGS "$SRC_FILE" > "$TMP_LOGFILE" 2>&1
		RETVAL=$?
		exec 2> /dev/tty

		if [ $MODE = "good" -a $RETVAL -ne 0 ]
		then
		    print_red "failed"
		    echo
		    echo "Expected the test to succeed, but it failed:"
		    exit 1
		elif [ $MODE = "bad" -a $RETVAL -eq 0 ]
		then
		    print_red "did not fail as expected"
		    echo
		    echo "Expected the test to fail, but it succeeded:" >&2
		    exit 1
		elif [ $MODE = "bad" -a $RETVAL -ne 0 ]
		then
		    print_green "failed as expected"
		else
		    print_green "success"
		fi
	    done

	if [ $? -ne 0 ]
	then
	    cat "$TMP_LOGFILE" >&2
	    exit 1
	fi
    done
done
