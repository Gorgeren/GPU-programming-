#! bin/bash

if test ! -d bin; then
    mkdir bin
fi

./$1 < $2
./topng bin/0.bin
