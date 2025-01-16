CC := gcc
CFLAGS := -lm
SOURCE := test_lstm.c weights.h dropbear.h mydumblittleplotroutine.h

.PHONY: all clean

all: test_lstm test_lstm_approx

test_lstm: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -o $@

test_lstm_approx: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -DUSE_APPROX_ACTIV_FNS -o $@

clean:
	-rm test_lstm
	-rm test_lstm_approx
