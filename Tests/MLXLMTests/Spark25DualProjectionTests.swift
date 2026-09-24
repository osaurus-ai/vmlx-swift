import Testing
import MLX
import MLXNN
@testable import MLXLLM

@Suite(.serialized)
struct Spark25DualProjectionTests {
    func layer(seed: UInt64) -> QuantizedLinear {
        MLXRandom.seed(seed)
        return QuantizedLinear(weight: (MLXRandom.normal([10240,2560])*0.02).asType(.bfloat16), bias:nil,groupSize:64,bits:6)
    }
    @Test func nativeQ6ExactAndStrided() {
        let g=layer(seed:7),u=layer(seed:19)
        eval(g.parameters(),u.parameters())
        for seed:UInt64 in [7,19,31] {
            MLXRandom.seed(seed)
            let base=MLXRandom.normal([1,1,5120]).asType(.bfloat16)
            for x in [base[0...,0...,.stride(by:2)],base[0...,0...,.stride(by:-2)]] {
                let expected=gelu(g(x))*u(x)
                let actual=Spark25DualProjection.apply(x,gate:g,up:u)
                #expect(actual != nil)
                if let actual { eval(expected,actual);#expect((expected .== actual).all().item(Bool.self)) }
            }
        }
    }
    @Test func fallbackContracts() {
        let g=layer(seed:11),u=layer(seed:13)
        let x=zeros([1,1,2560],dtype:.bfloat16)
        Stream.withNewDefaultStream(device:.cpu) {
            #expect(Spark25DualProjection.apply(x,gate:g,up:u)==nil)
        }
        for dtype:DType in [.float16,.float32] {
            #expect(Spark25DualProjection.apply(x.asType(dtype),gate:g,up:u)==nil)
        }
        for shape in [[1,2,2560],[2,1,2560],[1,2560],[1,1,1280]] {
            #expect(Spark25DualProjection.apply(zeros(shape,dtype:.bfloat16),gate:g,up:u)==nil)
        }
        let biased=QuantizedLinear(weight:g.weight,bias:zeros([10240],dtype:.bfloat16),scales:g.scales,biases:g.biases,groupSize:64,bits:6)
        #expect(Spark25DualProjection.apply(x,gate:biased,up:u)==nil)
        let mixed=QuantizedLinear(weight:g.weight,scales:g.scales.asType(.float16),biases:g.biases?.asType(.float16),groupSize:64,bits:6)
        #expect(Spark25DualProjection.apply(x,gate:mixed,up:u)==nil)
    }
    @Test func compiledFallbackParity() {
        let g=layer(seed:41),u=layer(seed:43)
        let x=MLXRandom.normal([1,1,2560]).asType(.bfloat16)
        let f=compile { (x:MLXArray) in Spark25DualProjection.apply(x,gate:g,up:u) ?? gelu(g(x))*u(x) }
        let expected=gelu(g(x))*u(x),actual=f(x)
        eval(expected,actual)
        #expect((expected .== actual).all().item(Bool.self))
    }
}
