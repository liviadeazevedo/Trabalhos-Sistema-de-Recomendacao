######################################################################################################################################
# Implementação do algoritmo K-nearest Neighbours
#
# Introdução a Sistemas de Recomendação - 2016/2
# Universidade Federal Rural do Rio de Janeiro
# Integrantes: Lívia de Azevedo, Gustavo Ebbo e Ivo Paiva
#
######################################################################################################################################

#Lendo a base de dados do MovieLens de 100K
data = readdlm("u.data")

#Com relação a base de dados
qtd_users = 943
qtd_itens = 1682

data = convert(Array{Int64},data)

#######################################################################################################################
#Funções das similaridades:

#Cosseno
#Correlação de Pearson
#Distância euclidiana
#Mean Squared Difference(Diferença quadrática média)

#"u" e "v": Id dos usuários que terão sua similaridade calculada.
function itens_em_comum(u,v)
   
    itens_u = find(r->r!=0,u)
    itens_v = find(r->r!=0,v)
    
    itens_uv = intersect(itens_u,itens_v)
    
    return itens_uv 
end

function cosseno(u,v)
    
    itens_uv = itens_em_comum(u,v)
    
    if(sum(u) == 0.0 || sum(v) == 0.0 || length(itens_uv) < 10)
        return -3
    end
    
    sum1 = sum(u[itens_uv] .* v[itens_uv])
    sum2 = sqrt(sum(u.^2) * sum(v.^2))
    
    if sum1 == 0
        return -3
    end
    
    return sum1 / sum2
end

function correlacao_pearson(u,v)
    
    itens_uv = itens_em_comum(u,v)
    
    mean_u = mean(u[itens_uv])
    mean_v = mean(v[itens_uv])
            
    if length(itens_uv) >= 10
        sum1 = sum((u[itens_uv] - mean_u) .* (v[itens_uv] - mean_v))
        sum2 = sqrt(sum((u[itens_uv] - mean_u).^2) * sum((v[itens_uv] - mean_v).^2))
                   
        return sum1 / sum2 
    else
        return -3
    end
end

function distancia_euclidiana(u,v)

    itens_uv = itens_em_comum(u,v)
    
    if length(itens_uv) >= 10
        sum1 = sqrt(sum((u[itens_uv] - v[itens_uv]).^2))
        
        return 1 / (1 + sum1)
    else
        return -3
    end
    
end

function diferenca_quadratica_media(u,v)

    itens_uv = itens_em_comum(u,v)
    
    if length(itens_uv) >= 10
        sum1 = sum((u[itens_uv] - v[itens_uv]).^2)
        sum1 = length(itens_uv) / sum1
   
        return 1 / (1 + sum1)
    else
        return -3
    end
end

#Função que embaralha o conjunto de dados.
function shuffle_data(data::AbstractArray)
    seed = convert(Int64,round(rand()*10000+1)) #gera seed
    rows_sort = shuffle(MersenneTwister(seed),collect(1:size(data)[1]))
    original_rows = collect(1:size(data)[1])
    data[original_rows,:] = data[rows_sort,:]
end

#######################################################################################################################
#similaridades_u: vetor que contem as similaridades dos usuários com relação a "u"
#"u" e "i": Números inteiros que correspondem ao usuário e ao item.
#notas: Matriz de notas usuário x item.
#k: Parâmetro do algoritmo dos vizinhos mais próximos.
function previsao(u,i,similaridades_u,notas,k)
    
    ########################Determinando os vizinhos mais próximos################################
        
    k_vizinhos_mais_proximos = zeros(k)
    k_vizinhos_mais_proximos = convert(Array{Int64},k_vizinhos_mais_proximos)
    
    usuarios_disp_aux = copy(similaridades_u)
    usuarios_disp_aux[u] = -3
    usuarios_disp_aux[find(r->r==0,notas[:,i])] = -3
    usuarios_disp_aux[find(r->r==0.0,similaridades_u)] = -3
    
    if length(find(r->r!=-3,usuarios_disp_aux)) >= k
        for j=1:k
            next_viz = indmax(usuarios_disp_aux)
            k_vizinhos_mais_proximos[j] = next_viz
            usuarios_disp_aux[next_viz] = -3
        end
    else
        return -1 #Não conseguiu prever para este valor de k.
    end
    ########################################################################################
    
    ##############################Realizando a previsão da nota#############################################
    sum1 = 0.0
    sum2 = 0.0

    for v in k_vizinhos_mais_proximos

        #Z-score(função de normalização)
        desvio_padrao_v = std(notas[v,find(r->r!=0,notas[v,:])])
        media_v = mean(notas[v,find(r->r!=0,notas[v,:])])
        Z_score = (notas[v,i] - media_v) / desvio_padrao_v

        sum1 += similaridades_u[v] * Z_score
        sum2 += abs(similaridades_u[v])
    end

    desvio_padrao_u = std(notas[u,find(r->r!=0,notas[u,:])])
    media_u = mean(notas[u,find(r->r!=0,notas[u,:])])
    
    previsao_ui = media_u + (desvio_padrao_u * (sum1 / sum2))

    return previsao_ui
    ########################################################################################################
end

#Função que calcula o MAE.
function mean_absolute_error(test,similaridades,notas,k)
    MAE = 0.0
    qtd_nao_previstos = 0
         
    for i=1:size(test)[1]
        
        prev = previsao(test[i,1],test[i,2],similaridades[test[i,1],:],notas,k)
                
        if prev != -1
            MAE += abs(test[i,3] - prev)
        else
            qtd_nao_previstos += 1
        end
    end
        
    MAE /= (size(test)[1] - qtd_nao_previstos)
    
    cobertura = qtd_nao_previstos
    
    return MAE,cobertura
end
#######################################################################################################################
#funcao_sim: Função de similaridade a ser usada.
function algoritmo_k_vizinhos_mais_proximos(data,k,funcao_sim,k_fold)
        
    if size(data)[1] % k_fold == 0
        
        #Criação da matriz de similaridade usuários x usuários
        similaridades = eye(qtd_users,qtd_users)
        
        num_elem_per_fold = convert(Int64,size(data)[1] / k_fold)
        train_size = size(data)[1] - num_elem_per_fold
        test_size = num_elem_per_fold
        data_index = collect(1:size(data)[1])
        fold_turn = 1
        finding_MAE = zeros(k_fold)
        qtd_nao_previstos_per_fold = zeros(k_fold)
        qtd_nao_previstos_per_fold = convert(Array{Int64},qtd_nao_previstos_per_fold)
        
        #Definindo o conjunto de treinamento e teste de acordo com o k-FCV.
        test = zeros(test_size,3)
        test = convert(Array{Int64},test)
        train = zeros(train_size,3)
        train = convert(Array{Int64},train)
        
        for iter=1:k_fold
            
            fold_index = collect(fold_turn:(iter * num_elem_per_fold))
            
            select_test = fold_index
            select_train = find(r->!(r in fold_index),data_index)

            train[1:train_size,1:3] = data[select_train[1:train_size],1:3]
            test[1:test_size,1:3] = data[select_test[1:test_size],1:3]
            
            
            #########################Execução do algoritmo em si#######################################
            
            #Criação da matriz items x usuários
            notas = zeros(qtd_users,qtd_itens)
            notas = convert(Array{Int64},notas)
            
            for i=1:size(train)[1]
                notas[train[i,1],train[i,2]] = train[i,3]
            end
            
            for i=2:qtd_users
                for j=1:(i-1)
                    similaridades[i,j] = funcao_sim(notas[i,:], notas[j,:])
                    similaridades[j,i] = similaridades[i,j]
                end
            end
            
            #Previsão do conjunto de teste
            aux_tupla = mean_absolute_error(test,similaridades,notas,k)
            finding_MAE[iter] = aux_tupla[1]
                        
            qtd_nao_previstos_per_fold[iter] = aux_tupla[2]
            ########################################################################################
            
            fold_turn += num_elem_per_fold
           
        end
        
        MAE_global = mean(finding_MAE)
        proporcao_cobertura = 100.0 - (100.0 * (sum(qtd_nao_previstos_per_fold) / (k_fold * test_size)))
        desvio_padrao_cobertura = std(qtd_nao_previstos_per_fold)  
    else
        return "Escolha um número múltiplo do tamanho do conjunto de dados."
    end
    
    return MAE_global,proporcao_cobertura,desvio_padrao_cobertura
end

#######################################################################################################################
#Realização dos testes(postos no relatório)
i = 10
k_fold = 5

while i <= 50
    
    println(i)
    
    println("Cosseno:")
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,cosseno,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,cosseno,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,cosseno,k_fold))
    
    println("Pearson:")
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,correlacao_pearson,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,correlacao_pearson,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,correlacao_pearson,k_fold))

    println("Distancia Euclidiana:")
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,distancia_euclidiana,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,distancia_euclidiana,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,distancia_euclidiana,k_fold))

    println("Diferença Quadrática Média:")
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,diferenca_quadratica_media,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,diferenca_quadratica_media,k_fold))
    shuffle_data(data)
    println(algoritmo_k_vizinhos_mais_proximos(data,i,diferenca_quadratica_media,k_fold))

    i = i + 10    
end