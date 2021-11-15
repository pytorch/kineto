# How to generate the api.ts

## Prerequisites
1. install java
2. run command
```bash
  cd fe
  wget https://repo1.maven.org/maven2/io/swagger/codegen/v3/swagger-codegen-cli/3.0.25/swagger-codegen-cli-3.0.25.jar -O swagger-codegen-cli.jar
  java -jar swagger-codegen-cli.jar generate -i ./src/api/openapi.yaml -l typescript-fetch -o ./src/api/generated/  --additional-properties modelPropertyNaming=original
  rm ./src/api/generated/api_test.spec.ts
  yarn prettier --end-of-line lf
  python ./scripts/add_header.py ./src/api/generated/
```
