#!/bin/bash
set -e

current_dir="$( cd "$( dirname "$0" )" && pwd )"
FE_ROOT="$(dirname "$current_dir")"
cd $FE_ROOT/

java -jar $FE_ROOT/swagger-codegen-cli.jar generate -i $FE_ROOT/src/api/openapi.yaml -l typescript-fetch -o $FE_ROOT/src/api/generated/ --additional-properties modelPropertyNaming=original
rm $FE_ROOT/src/api/generated/api_test.spec.ts
yarn prettier --end-of-line lf
python $FE_ROOT/scripts/add_header.py $FE_ROOT/src/api/generated/

yarn build:copy
