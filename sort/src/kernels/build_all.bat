REM Currently only compile simple store kernels.
REM Transaction lists are still experimental.

CALL count.bat
CALL hist_simple.bat

CALL sort_128_8_index_simple.bat
CALL sort_128_8_key_simple.bat
CALL sort_128_8_single_simple.bat
CALL sort_128_8_multi_simple.bat

CALL sort_256_8_index_simple.bat
CALL sort_256_8_key_simple.bat
CALL sort_256_8_single_simple.bat
CALL sort_256_8_multi_simple.bat
