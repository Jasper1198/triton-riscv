import sys
from pathlib import Path

import triton

ROOT = Path("/workspace/triton-riscv")
sys.path.insert(0, str(ROOT / "python/evo_vendor"))
sys.path.insert(0, str(ROOT / "python/examples"))

from dev.search_plan.evo_search import EvoAutotuner
from rvv_vec_add_elf import vector_add_kernel
from triton.backends.triton_shared.riscv import (
    compile_kernel_to_elf,
    run_qemu,
)


def main():
    # 先測試候選生成與執行流程，尚未啟動演化搜尋。
    seeds = [
        triton.Config(
            {"BLOCK_SIZE": size},
            num_warps=1,
            num_stages=1,
            num_ctas=1,
        )
        for size in [8, 64]
    ]

    tuner = EvoAutotuner(
        fn=None,
        arg_names=[],
        configs=seeds,
        key=[],
        pop_size=4,
        warmup_configs=0,
        max_tuned_configs=4,
        cost_model="random",
        do_bench=lambda *args, **kwargs: None,
    )
    tuner.initialize_population()

    # 故意使用非 tile 整數倍，順便驗證邊界遮罩。
    size = 67
    x = [float(i) for i in range(size)]
    y = [float(2 * i) for i in range(size)]
    expected = [a + b for a, b in zip(x, y)]

    output_dir = ROOT / "artifacts/riscv/evo-validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    for chromosome in tuner.population:
        config = chromosome.to_config()
        block_size = config.kwargs["BLOCK_SIZE"]
        elf = output_dir / f"vecadd-block-{block_size}.elf"

        print(f"\n編譯 BLOCK_SIZE={block_size}", flush=True)

        # 目前只把 tile 參數傳給 ELF 編譯介面。
        compile_kernel_to_elf(
            vector_add_kernel,
            arguments={
                "x_ptr": x,
                "y_ptr": y,
                "output_ptr": [0.0] * size,
                "n_elements": size,
            },
            constexprs={"BLOCK_SIZE": block_size},
            grid=(triton.cdiv(size, block_size),),
            output=elf,
            expected={"output_ptr": expected},
            atol=0.0,
        )

        run_qemu(elf)
        passed += 1
        print(f"候選通過：BLOCK_SIZE={block_size}", flush=True)

    print(f"\n全部通過：{passed} 個候選")


if __name__ == "__main__":
    main()
