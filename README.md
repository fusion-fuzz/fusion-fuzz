# Fusion Fuzz

![Tests](https://github.com/fusion-fuzz/fusion-fuzz/actions/workflows/tests.yml/badge.svg)

Fusion-fuzz is a scalable and effective fuzzer to discover bugs in various compilers and interpreters.

The core idea of fusion-fuzz is **program fusion**, which bridges the behavior of two (or more) independent seed programs so the fused program exercises interactions neither seed triggers alone.

Program fusion now has three fusion strategies: **dataflow fusion**, **state fusion**, and **declaration fusion**.

- **Dataflow fusion** - connect dataflow from parent seeds by interleaving variables.

Example of dataflow fusion:

```php
/* seed A */
$dom = new DOMDocument;
$dom->loadXML(..);
$ref = $dom->documentElement->firstChild;
$nodes = $ref->childNodes;

$fusion = $nodes; // dataflow bridge 

/* seed B */
$values = array(..);
foreach ($fusion as $str) //foreach ($values as $str)
    { $enc = base64_encode($str); }

/* output */
// AddressSanitizer: heap-use-after-free ..
```

- **State fusion** — bridge behaviors at the *interesting program points - the point holding more live variables* via interleaving program statements.

```php
<?php
function dump($dom, $name) {
$list = $dom->getElementsByTagName($name)[0]
->getInScopeNamespaces();
foreach ($list as $entry) {
	/* self state fusion */ /* more live variables */
	$dom = Dom\XMLDocument::createFromString(<<<XML
	<root xmlns="urn:a">
	<child xmlns="">
	<c:child xmlns:c="urn:c"/> </child>
	<b:sibling xmlns:b="urn:b" xmlns:d="urn:d" d:foo="bar">
	</b:sibling>
	</root>
	XML);
	dump($dom, 'c:child');
	dump($dom, 'child');
}
}
$dom = Dom\XMLDocument::createFromString(<<<XML
<root xmlns="urn:a">
<child xmlns="">
<c:child xmlns:c="urn:c"/> </child>
<b:sibling xmlns:b="urn:b" xmlns:d="urn:d" d:foo="bar">
<d:child xmlns:d="urn:d2"/>
</b:sibling>
</root>
XML);
dump($dom, 'c:child');
dump($dom, 'child');
// SUMMARY: AddressSanitizer: double-free
```

- **Declaration fusion** — enforce *declaration dependencies* instead of variables or statements. 

Example of declaration fusion:

```swift
/* seed A */
class C: P {}
/* seed B */ 
class Generic<T> : Concrete {
  typealias GenericAlias = (T, T)
}
protocol BaseProto: C {} /* declaration dependency from A to B */
protocol ProtoRefinesClass where Self : Generic<Int>, Self : BaseProto {
  func requirementUsesClassTypes(_: ConcreteAlias, _: GenericAlias)
}
```

The scalability of fusion-fuzz is mostly from **seed migration**, which translates seed programs from one target into every other target. 

Supported projects are:

| Project | Status | Dataflow fusion | State fusion | Declaration fusion |
|---------|--------|:---:|:---:|:---:|
| ![PHP](https://img.shields.io/badge/PHP-supported-brightgreen?logo=php&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![CPython](https://img.shields.io/badge/CPython-supported-brightgreen?logo=python&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Swift](https://img.shields.io/badge/Swift-supported-brightgreen?logo=swift&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Clang](https://img.shields.io/badge/Clang-supported-brightgreen?logo=llvm&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![MLIR](https://img.shields.io/badge/MLIR-supported-brightgreen?logo=llvm&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Flang](https://img.shields.io/badge/Flang-supported-brightgreen?logo=llvm&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![LFortran](https://img.shields.io/badge/LFortran-supported-brightgreen?logo=fortran&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Haskell](https://img.shields.io/badge/GHC-supported-brightgreen?logo=haskell&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Rust](https://img.shields.io/badge/Rust-experimental-orange?logo=rust&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![GCC](https://img.shields.io/badge/GCC-experimental-orange?logo=gnu&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![Go](https://img.shields.io/badge/Go-experimental-orange?logo=go&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![V8](https://img.shields.io/badge/V8-experimental-orange?logo=googlechrome&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![SpiderMonkey](https://img.shields.io/badge/SpiderMonkey-experimental-orange?logo=firefoxbrowser&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![Naga](https://img.shields.io/badge/Naga-experimental-orange?logo=webgpu&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![Tint](https://img.shields.io/badge/Tint-experimental-orange?logo=webgpu&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |
| ![Triton](https://img.shields.io/badge/Triton-experimental-orange?logo=nvidia&logoColor=white) | **Experimental** | ✅ | ✅ | ✅ |

**Supported** means the adapter has been run at length, its valid-fusion rate
measured, and real bugs reported from it. **Experimental** means all three
fusion strategies are wired and the adapter runs end to end, but it has had far
less fuzzing time, so its valid-fusion rate and its oracle's false-positive
behaviour are not yet well characterised.

**Bugs found by Fusion Fuzz are tracked at https://fusion-fuzz.github.io (updated periodically).**

Try fusion-fuzz: 

(fusion-fuzz is tested in ubuntu 22.04 and ubuntu 24.04)

(i) before cloning fusion-fuzz, please install *git-lfs*, which is necessary to download our translated corpus (hundreds of MBs).

(ii) install *docker.io*. fusion-fuzz only runs in the docker to keep the host clean.

(iii) for every supported project, go to the project folder. Take PHP as example: `cd ./projects/php` and build the docker `docker build -t fusion-fuzz-php .`

(iv) start the docker `docker run --name fuzz-php -dit -v <your fusion-fuzz path>:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-php:latest` and go to the docker bash `docker exec -it fuzz-php bash`

(v) in the docker bash, you might want to grant 777 permissions for the fuzzing folder first. fusion-fuzz docker user:pass is `fuzz:fuzz`. You can try `sudo chmod -R 777 /home/fuzz/WorkSpace/fusion-fuzz`

(vi) create a tmux bash and start the fuzzer in the docker. `python3 main.py --project php --setup --pre-analysis --dataflow-fusion --state-fusion --declaration-fusion`


You should be able to see bugs found by fusion-fuzz in ./output/bugs/<project-name>.
