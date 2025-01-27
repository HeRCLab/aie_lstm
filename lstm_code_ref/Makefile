CC := gcc
CFLAGS := -lm
SOURCE := test_lstm.c weights.h dropbear.h mydumblittleplotroutine.h

.PHONY: all clean

all: test_lstm test_lstm_approx test_lstm_pwl

test_lstm: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -o $@

test_lstm_approx: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -DUSE_APPROX_ACTIVATION_FNS -o $@

test_lstm_pwl: $(SOURCE)
	$(CC) test_lstm.c $(CFLAGS) -DUSE_PWL_ACTIVATION_FNS -o $@

clean:
	-rm test_lstm
	-rm test_lstm_approx
