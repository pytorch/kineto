#!/bin/bash
set -e

current_dir="$( cd "$( dirname "$0" )" && pwd )"
FE_ROOT="$(dirname "$current_dir")"

# # install nodejs
if ! command -v node &> /dev/null
then
    curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# install yarn
if ! command -v yarn &> /dev/null
then
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
    sudo apt update && sudo apt install yarn
fi

# download swagger-codegen-cli
if [[ ! -f "$FE_ROOT/swagger-codegen-cli.jar" ]]; then
    wget https://repo1.maven.org/maven2/io/swagger/codegen/v3/swagger-codegen-cli/3.0.25/swagger-codegen-cli-3.0.25.jar -O swagger-codegen-cli.jar
fi
