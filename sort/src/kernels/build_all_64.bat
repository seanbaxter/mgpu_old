REM Currently only compile simple store kernels.
REM Transaction lists are still experimental.

CALL count_64.bat
CALL hist_simple_64.bat

CALL sort_128_8_index_simple_64.bat
CALL sort_128_8_key_simple_64.bat
CALL sort_128_8_single_simple_64.bat
CALL sort_128_8_multi_simple_64.bat

CALL sort_256_8_index_simple_64.bat
CALL sort_256_8_key_simple_64.bat
CALL sort_256_8_single_simple_64.bat
CALL sort_256_8_multi_simple_64.bat
