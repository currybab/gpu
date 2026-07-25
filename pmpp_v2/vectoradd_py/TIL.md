### pytorch custom cuda kernel 호출시 `reinterpret_cast<__half*>(out.data_ptr<at::Half>())`가 필요한 이유.

PyTorch의 fp16 타입(at::Half)과 CUDA의 fp16 타입(__half)이 서로 다른 타입이라서.
at::Half는 PyTorch가 CPU/GPU 공용으로 쓰려고 만든, uint16_t 하나를 감싼 구조체.
__half는 NVIDIA의 cuda_fp16.h에 정의된 CUDA 네이티브 타입.
둘은 메모리 레이아웃이 완전히 같지만(둘 다 16비트), C++ 입장에선 아무 관계 없는 별개 타입이라 포인터가 암묵적으로 변환되지 않음.

### GPU 메모리 접근 정리: 캐시라인, 섹터, 코얼레싱, 벡터 폭

캐시라인은 128 Byte이고 실제 전송 단위는 32 Byte 섹터다. 효율은 요청 바이트 ÷ 페치 바이트이고, 하한은 ceil(요청 바이트 / 32) 섹터다. 워프의 주소가 연속이고 정렬돼 있으면 스레드당 폭이 2B든 4B든 16B든 효율은 100%다. 스레드당 최대 폭은 128비트이며, 이때 워프의 요청은 512Byte로 섹터 16개(라인 4개)에 걸친다. 폭을 넓혀도 대역폭 효율은 그대로이고, 줄어드는 것은 명령 수다.

### triton에서 BLOCK_SIZE 정하기 (vectoradd 예제 기준)

스레드당 128bit 로드를 가정했을 때, `__half` 타입은 8개가 로드됨.
`num_warps`를 기본값인 4로 사용하면, `BLOCK_SIZE`는 8 * 4 * 32 = 1024가 되어야 함.
`num_warps`를 8로 설정하면, `BLOCK_SIZE`는 8 * 8 * 32 = 2048가 되어야 함.
