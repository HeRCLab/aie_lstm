CC := gcc
CFLAGS := -lm
SOURCE := test_lstm.c weights.h dropbear.h mydumblittleplotroutine.h

test_lstm: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -o $@

test_lstm_approx: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -DUSE_APPROX_SIGM -o $@
