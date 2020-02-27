#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

print_green() {
    [ -t 1 ] && echo -e "\e[32m$@\e[39m" || echo "$@"
}

print_yellow() {
    [ -t 1 ] && echo -e "\e[93m$@\e[39m" || echo "$@"
}

print_red() {
    [ -t 1 ] && echo -e "\e[31m$@\e[39m" || echo "$@"
}

export TECKYL="$PWD/bin/teckyl"
export LLC="$PWD/llvm-project/llvm/bin/llc"
export MLIR_OPT="$PWD/llvm-project/llvm/bin/mlir-opt"
export MLIR_TRANSLATE="$PWD/llvm-project/llvm/bin/mlir-translate"

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

if [ -x "$MLIR_OPT" -a -x "$MLIR_TRANSLATE" ]
then
    for TEST_DIR in "$BASE_DIR"/tests/exec/*
    do
	TEST_BASE=$(basename "$TEST_DIR")

	if [ "$TEST_BASE" != "lib" ]
	then
	    printf "Running execution test %s... " "$TEST_BASE"

	    mkdir -p "tests/exec/$TEST_BASE" || \
		die "Could not create test directory"

	    export BUILDDIR="$PWD/tests/exec/$TEST_BASE"

	    make -C "$TEST_DIR" > "$TMP_LOGFILE" 2>&1

	    if [ $? -ne 0 ]
	    then
	        print_red "build failed."
		echo
		cat "$TMP_LOGFILE" >&2
		exit 1
	    fi

	    make -C "$TEST_DIR" run > "$TMP_LOGFILE" 2>&1

	    if [ $? -ne 0 ]
	    then
	        print_red "failed."
		echo
		cat "$TMP_LOGFILE" >&2
		exit 1
	    else
		print_green "success"
	    fi
	fi
    done
else
    print_yellow "Binaries mlir-opt and mlir-translate haven't been built." \
		 "Skipping execution tests."
fi
