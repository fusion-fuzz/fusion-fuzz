struct S {};
S f() {
  __constant__ static S s;
  return s;
}
void g() { f(); }
