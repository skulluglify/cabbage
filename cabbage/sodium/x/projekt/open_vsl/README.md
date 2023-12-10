# OpenVSL (Virtual Script Language)
## Like EcmaScript, some methods

```txt
vFunction()
vFunction.apply()

vFunctionMember.bind
vFunctionMember.call()
vFunctionMember.args
vFunctionMember.kwargs

// AOT (secure)
main.vsl -> main.vmo (evaluation with dumped VMO) -> main.psock (transport as network stream)
main.psock -> main.vmo -> execution (receiving from network stream)

// JIT (fast)
main.vsl -> main.websock (transport as network stream)
main.websock -> evaluation (receiving from network stream)

// archive scripts and assets
main.vmo (nosign, excludes metadata.info)
main.vpk (signatures, includes magicnum.ext, compressed by archive)
```

```txt
var obj const = object { a: 12, b: 13, c: 14 }

for each item and index from obj do
  print index, item.key, item.value
done

print obj as array
```

```txt
// main run script.
runtime/bin/vsl-run main.vsl

// added modules vsl sources.
tools/bin/vsl-dump -o main.vmo -p modules/vsl main.vsl

// compilation and compressed by archive.
tools/bin/vpk-comp -o example.vpk example/vsl/main.vsl

// added modules vsl libraries.
runtime/bin/vexec -p modules/vpks main.vmo

// run compilation archive.
runtime/bin/vpk-run example.vpk
```
