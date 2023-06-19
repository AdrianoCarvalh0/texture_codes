import sys
sys.path.insert(0, "/home/adriano/projeto_mestrado/modules")

import numpy as np
import pickle
import vessel_statistics as vs

if __name__ == '__main__':
    pasta_mestrado ='/home/adriano/projeto_mestrado/modules'

    dir = f'{pasta_mestrado}/Vessel_Models_pickle'
   
    #coloca nas varíaveis nom e tam, os nomes dos arquivos que existem no diretório e pega a quantidade de itens que existem
    nom, tam = vs.ready_directory(dir)

    maximum = []
    minimum = []
    vetor_diametros = []
    i=1
    #faz a varredura de todos os itens que estão no diretório
    for i in range(tam):  
        #pega o nome da imagem e armazena na variável local
        local = nom[i]
        #faz a carregamento do .pickle
        data_dump = pickle.load(open(local,"rb"))
        #absorve os três índices que existem no .pickle
        vessel_model = data_dump['vessel_model']
        first_point = data_dump['primeiro_ponto']
        img_file = data_dump['img_file']  

        #instanciação da variável vessel_map
        #o mapa do vaso possui vários atributos que serão utilizados para chamar as funções
        vessel_map = vessel_model.vessel_map 
        
        #pega a metade inteira do tamanho do vessel_map.mapped_values
        half_size_vessel_map = len(vessel_map.mapped_values)//2

        #variáveis que pegam o índice do menor e do menor valor da posição do meio
        max = np.argmax(vessel_map.mapped_values[half_size_vessel_map])
        min = np.argmin(vessel_map.mapped_values[half_size_vessel_map])  

        #variáveis que pegam o valor armazenados no menor e maior valor encontrado
        maximum.append(vessel_map.mapped_values[half_size_vessel_map][max])
        minimum.append(vessel_map.mapped_values[half_size_vessel_map][min])

        #retorna a média de todos os valores mapeados ao longo das linhas
        means = np.mean(vessel_map.mapped_values, axis=1)

        #retorna o desvio padrão de todos os valores mapeados ao longo das linhas
        std_dev = np.std(vessel_map.mapped_values, axis=1)  

        #retorna o desvio padrão de todos os valores mapeados ao longo das colunas
        std_dev2 = np.std(vessel_map.mapped_values, axis=0)    

        #plot of the vessel map, min value is 0 and max value is 60
        vs.plot_vessel_map(vessel_map) 

        #plot do recorte
        vs.plot_clipping(vessel_map)
        
        #plot da intensidade das linhas interemediária, uma acima e uma abaixo
        vs.plot_intensity_lines(vessel_map, half_size_vessel_map) 

        #plot mostrando a diferença entre a média e o desvio padrão
        vs.plot_fill_means_std_dev(means, std_dev)

        #tentativa de plotar a diferença entre o desvio padrão com a coluna normalizada
        #plot_fill_means_std_dev_normal(intensity_cols_values_all[0], std_dev2)
        
        #plota o diâmetro do vaso
        vs.plot_diameter_vessel(vessel_map)  

        #plota a intensidade das colunas, exibe onde começa e termina o vaso
        vs.plot_intensity_cols_with_line_vessel(vessel_map)

        #plota a intensidade das colunas normalizadas com a linha do centro (metade dos índices das colunas), 
        #exibe onde começa e termina o vaso, retirando a dependência de se começar do ponto (0,0)
        vs.plot_intensity_cols_with_line_vessel_normal(vessel_map)

        #plota todas as intensidades normalizadas
        intensities_common_axis, l2_chapeu_axis = vs.return_all_instisitys_normal(vessel_map)

        vs.plot_all_intensities_columns(intensities_common_axis, l2_chapeu_axis)

        #plot_fill_means_std_dev_normal_all(intensities_common_axis)

        #plota a diferença entre os máximos e mínimos de todas as extrações
        vs.plot_min_max_medial_line(minimum,maximum)

        #VER plt.xaxis.set_Visible(False)

        #usar o inkscape - software de imagens


