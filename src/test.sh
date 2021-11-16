#!/bin/bash

#Accesses to Global Memory
./non_tiled_matmul 32 32 32 32
echo "##################"

./non_tiled_matmul 256 256 256 256
echo "##################"

./non_tiled_matmul 4096 2048 2048 4096
echo "##################"

#Accesses to Shared Memory
./tiled_matmul 32 32 32 32
echo "##################"

./tiled_matmul 256 256 256 256
echo "##################"

./tiled_matmul 4096 2048 2048 4096
