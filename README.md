# land-report

```bash
        routines_input_file_name = f'azure://planetary-routines-input/MONITORAMENTO_SICOR_BACEN/{file}'

        routines_output_name_space = f'azure://planetary-routines-output/MONITORAMENTO_SICOR_BACEN/{os.path.splitext(file)[0]}.pdf'

/routine/.venv/bin/python -u routine.py 'azure://planetary-routines-input/MONITORAMENTO_SICOR_BACEN/20250862734_Monte Alegre.xml' 'azure://planetary-routines-output/MONITORAMENTO_SICOR_BACEN/20250862734_Monte Alegre.pdf'
```

20250862734_Monte Alegre.xml 20241906129_Bom Jesus.xml 20241906282_Special Fruit.xml 20241938695_Impact.xml 20241939101_Marcos Beck.xml 20241939150_Marcos Beck.xml 20250002784_Bom Jesus.xml 20250003580_São Martinho.xml 20250003988_PS Agro.xml 20250020768_Darci Potrich.xml 20250783561_Darci Potrich.xml 20250783755_PS Agro.xml 20250784086_Horita.xml 20250786034_Agro Trojan.xml


- [X] Tabela de coordenadas geográficas.

- [X] Tamanho do plot com as imagens de satélite.

- [X] Número de CARs distintos no report inicial.

- [] Plot com a quantidade de área em hectares de cada cluster.

- [] 

- [X] Melhorar a saturação da imagem RGB.

- [X] Tirar a compacidade dos mapas.

- [X] Deixar menos pesado o filtro savitzgolay no EVI.

- [] Falar do satélite e do sensor, resoluções espectral e etc...