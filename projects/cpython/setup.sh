git clone https://github.com/python/cpython.git
cd ./cpython; mkdir build; cd build; CC=clang-21 CXX=clang++-21 ../configure --with-pydebug --enable-experimental-jit=yes --with-address-sanitizer; #--with-thread-sanitizer # --with-undefined-behavior-sanitizer --with-thread-sanitizer; make -j16 --with-pydebug
make -j16
