#!/bin/bash

# Define os caminhos base para facilitar a leitura e manutenção
BASE_INPUT="azure://planetary-routines-input/MONITORAMENTO_SICOR_BACEN/"
BASE_OUTPUT="azure://planetary-routines-output/MONITORAMENTO_SICOR_BACEN/"

# Lista de arquivos (Array) para garantir que espaços nos nomes sejam respeitados
files=(
    #"20250862734_Monte Alegre.xml"
    #"20241906129_ Bom Jesus.xml"
    #"20241906282_Special Fruit.xml"
    #"20241938695_Impact.xml"
    #"20241939101_Marcos Beck.xml"
    #"20241939150_Marcos Beck.xml"
    #"20250002784_ Bom Jesus.xml"
    #"20250003580_São Martinho.xml"
    #"20250003988_PS Agro.xml"
    "20250020768_Darci Potrich.xml"
    #"20250783561_Darci Potrich.xml"
    #"20250783755_PS Agro.xml"
    "20250784086_Horita.xml"
    #"20250786034_Agro Trojan.xml"
)

# Loop para iterar sobre cada arquivo
for file_xml in "${files[@]}"; do
    
    # Substitui a extensão .xml por .pdf
    file_pdf="${file_xml%.xml}.pdf"
    
    echo "---------------------------------------------------"
    echo "Processando: $file_xml"
    
    # Monta e executa o comando
    # As aspas nos caminhos "${...}" são vitais para funcionar com os espaços
    python -u routine.py "${BASE_INPUT}${file_xml}" "${BASE_OUTPUT}${file_pdf}"
    
    # Verifica se o comando deu erro (opcional, mas recomendado)
    if [ $? -eq 0 ]; then
        echo "Sucesso: $file_pdf gerado."
    else
        echo "ERRO ao processar: $file_xml"
    fi

done
