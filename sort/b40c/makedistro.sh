#!/bin/bash

export VER=`grep -o v[0-9]*.[0-9]*.[0-9]* VERSION.TXT`

tar --exclude=*/.svn --exclude=*/bin --exclude=*/testlab -czvf B40C_LsbRadixSort.$VER.tgz    LICENSE.TXT VERSION.TXT Common ExamplesInclude LsbRadixSort
tar --exclude=*/.svn --exclude=*/bin --exclude=*/testlab -czvf B40C_BFS.$VER.tgz             LICENSE.TXT VERSION.TXT Common ExamplesInclude BFS
tar --exclude=*/.svn --exclude=*/bin --exclude=*/testlab -czvf B40C_AllPrimitives.$VER.tgz      LICENSE.TXT VERSION.TXT Common ExamplesInclude LsbRadixSort BFS
