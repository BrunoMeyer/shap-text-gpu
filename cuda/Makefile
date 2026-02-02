NVCC ?= nvcc
CXXFLAGS ?= -O2 -std=c++17

BIN_DIR := ../out

all: $(BIN_DIR)/feedforward

$(BIN_DIR)/feedforward: feedforward.cu | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -f $(BIN_DIR)/feedforward
