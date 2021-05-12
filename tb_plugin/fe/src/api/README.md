# How to generate the api.ts

## Prerequisites
1. wget https://repo1.maven.org/maven2/io/swagger/codegen/v3/swagger-codegen-cli/3.0.25/swagger-codegen-cli-3.0.25.jar -O swagger-codegen-cli.jar
2. install java
3. cd fe/src/api
3. java -jar swagger-codegen-cli.jar generate -i openapi.yaml -l typescript-fetch -o ./generated/
