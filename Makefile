.PHONY: benchmark bench-report

benchmark:
	cd core && python -m benchmarks.runner

bench-report:
	cd core && python -m benchmarks.report
