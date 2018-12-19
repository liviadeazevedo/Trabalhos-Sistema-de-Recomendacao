######################################################################################################################################
# Trabalho Final: Análise do Ruído Natural na Filtragem Colaborativa
#
# Introdução a Sistemas de Recomendação - 2016/2
# Universidade Federal Rural do Rio de Janeiro
# Integrantes: Lívia de Azevedo, Gustavo Ebbo e Ivo Paiva
#
######################################################################################################################################


############################Construção das bases de treinamento do experimento##############################################################################
#Com relação a base de dados
qtd_users = 943
qtd_itens = 1682

#Lendo a base de dados do MovieLens de 100K
data = readdlm("u.data")

#Divisão aleatória da base de dados em teste e treinamento
training_size = 90000
test_size = 10000

train1 = zeros(training_size,5) #90000 elementos ------> 90% 
test1 = zeros(test_size,3) #10000 elementos --------> 10%

ordem_aleatoria = shuffle(MersenneTwister(4861),collect(1:size(data)[1]))

escolhas_treinamento = find(r->r, ordem_aleatoria .> test_size)
escolhas_teste = find(r->r, !(ordem_aleatoria .> test_size))

train1[1:training_size,1:4] = data[escolhas_treinamento[1:training_size],1:4]
test1[1:test_size,1:3] = data[escolhas_teste[1:test_size],1:3]

train2 = zeros(training_size,5) #90000 elementos ------> 90% 
test2 = zeros(test_size,3) #10000 elementos --------> 10%

ordem_aleatoria = shuffle(MersenneTwister(1278),collect(1:size(data)[1]))

escolhas_treinamento = find(r->r, ordem_aleatoria .> test_size)
escolhas_teste = find(r->r, !(ordem_aleatoria .> test_size))

train2[1:training_size,1:4] = data[escolhas_treinamento[1:training_size],1:4]
test2[1:test_size,1:3] = data[escolhas_teste[1:test_size],1:3]

train3 = zeros(training_size,5) #90000 elementos ------> 90% 
test3 = zeros(test_size,3) #10000 elementos --------> 10%

ordem_aleatoria = shuffle(MersenneTwister(7295),collect(1:size(data)[1]))

escolhas_treinamento = find(r->r, ordem_aleatoria .> test_size)
escolhas_teste = find(r->r, !(ordem_aleatoria .> test_size))

train3[1:training_size,1:4] = data[escolhas_treinamento[1:training_size],1:4]
test3[1:test_size,1:3] = data[escolhas_teste[1:test_size],1:3]

train4 = zeros(training_size,5) #90000 elementos ------> 90%
test4 = zeros(test_size,3) #10000 elementos --------> 10%

ordem_aleatoria = shuffle(MersenneTwister(821),collect(1:size(data)[1]))

escolhas_treinamento = find(r->r, ordem_aleatoria .> test_size)
escolhas_teste = find(r->r, !(ordem_aleatoria .> test_size))

train4[1:training_size,1:4] = data[escolhas_treinamento[1:training_size],1:4]
test4[1:test_size,1:3] = data[escolhas_teste[1:test_size],1:3]

train5 = zeros(training_size,5) #90000 elementos ------> 90%
test5 = zeros(test_size,3) #10000 elementos --------> 10%

ordem_aleatoria = shuffle(MersenneTwister(5569),collect(1:size(data)[1]))

escolhas_treinamento = find(r->r, ordem_aleatoria .> test_size)
escolhas_teste = find(r->r, !(ordem_aleatoria .> test_size))

train5[1:training_size,1:4] = data[escolhas_treinamento[1:training_size],1:4]
test5[1:test_size,1:3] = data[escolhas_teste[1:test_size],1:3]
######################################################################################################################################

############################Criando as bases ruidosas##############################################################################################
#Gerar ruído aleatório na base
function gerar_ruido(data,porcentagem_ruido)

    indices_novos_ruidosos = rand(1:size(data)[1],convert(Int64,round((size(data)[1] * porcentagem_ruido),RoundUp)))
    
    for i in indices_novos_ruidosos
        nota_ruido = data[i,3]
        
        while nota_ruido == data[i,3]
            nota_ruido = convert(Float64,rand(1:5)) #Gera aleatoriamente uma nota entre 1 e 5(inteira)
        end
        
        data[i,3] = nota_ruido
    end
end

#Inicializando as bases ruidosas
train1_ruido = copy(train1)
train2_ruido = copy(train2)
train3_ruido = copy(train3)
train4_ruido = copy(train4)
train5_ruido = copy(train5)

#30% dos elementos da base de treinamento(27000) conterão um ruído natural gerado artificialmente.
#gerar_ruido(train1_ruido,0.3)
#gerar_ruido(train2_ruido,0.3)
#gerar_ruido(train3_ruido,0.3)
#gerar_ruido(train4_ruido,0.3)
#gerar_ruido(train5_ruido,0.3)

#10% dos elementos
#gerar_ruido(train1_ruido,0.1)
#gerar_ruido(train2_ruido,0.1)
#gerar_ruido(train3_ruido,0.1)
#gerar_ruido(train4_ruido,0.1)
#gerar_ruido(train5_ruido,0.1)

#20% dos elementos.
#gerar_ruido(train1_ruido,0.2)
#gerar_ruido(train2_ruido,0.2)
#gerar_ruido(train3_ruido,0.2)
#gerar_ruido(train4_ruido,0.2)
#gerar_ruido(train5_ruido,0.2)
######################################################################################################################################

#######################################Funções para o KNN#################################################################################
function itens_em_comum(u,v)
   
    itens_u = find(r->(r!=0 && r!=-1),u)
    itens_v = find(r->(r!=0 && r!=-1),v)
    
    itens_uv = intersect(itens_u,itens_v)
    
    return itens_uv 
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
######################################################################################################################################

################################Implementação dos algoritmos de ruído###################################################

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
    
    if !(isnan(previsao_ui))
        return previsao_ui
    else
        return -1
    end
    ########################################################################################################
end

################################################Mahony####################################################################

# 1) Definir o KNN com Pearson e k = 35 para realizar a previsão da nota para o cálculo da consistência
# 2) Utilizar o valor de consistencia para determinar quem é o ruído, considerando apenas as notas dentro do training_set
#(utilizar a estratégia do leave-one-out);
# 3) Remover todos os possíveis ruídos da base de dados.

#train = conjunto de treinamento considerado
#limiar: valor limitante para verificação se uma nota possuiu ruído ou não
#min e max nota: nota mínima e máxima possível, respectivamente
#notas: Matriz de usuário x itens das notas
#similaridades: Matriz de similaridade obtida usando as notas do conjuto de treinamento.
function correcao_ruido_mahony(train,limiar,min_nota,max_nota,notas,similaridades)
        
    for i=1:size(train)[1]
        prev = previsao(convert(Int64,train[i,1]),convert(Int64,train[i,2]),similaridades[convert(Int64,train[i,1]),:],
               notas,35)
        if prev != -1
            consistencia = abs(train[i,3] - prev) / (max_nota - min_nota)
            
            if consistencia > limiar #Esta nota é um possível ruído. Devemos exclui-la.
                train[i,4] = -1
            end
        end
    end
end

function reset_mahony(data)
    data[:,4] = 80000
end
##########################################################################################################################

##############################################Toledo######################################################################
#train = conjunto de treinamento considerado
#min e max nota: nota mínima e máxima possível, respectivamente
#escala: Um array contendo os valores possíveis de notas(Ex: MovieLens é um array [1 2 3 4 5])
#notas: Matriz de usuário x itens das notas
#similaridades: Matriz de similaridade obtida usando as notas do conjuto de treinamento.
function correcao_ruido_toledo(train,min_nota,max_nota,escala,notas,similaridades)
    
    #######################Parte da detecção do ruído############################################################
        
    #Abordagem escolhida: global-pv(ku,vu,ki,vi,k,v)
    
    k = min_nota + round(( 1/3 ) * (max_nota - min_nota))
    v = max_nota - round(( 1/3 ) * (max_nota - min_nota))
    
    ku = k
    vu = v
    ki = k
    vi = v
        
    #Defnindo o limiar da perspactiva global(menor diferença entre os valores dentro da escala de valores possíveis)
    limiar = max_nota
    
    for i=1:(size(escala)[1] - 1)
        if limiar > (escala[i + 1,1] - escala[i,1])
            limiar = escala[i + 1,1] - escala[i,1]
        end
    end
        
    possiveis_ruidos = [] #Conjunto de índices dos possíveis ruídos.
    
    #Matrizes que contem os conjuntos classificatórios para usuário e item.
    #Colunas: 1 = W(u ou i), 2 = A(u ou i), 3 = S(u ou i), 4 = Classificação do usuário ou item.
    users_sets = convert(Array{Int64},zeros(qtd_users,4)) 
    itens_sets = convert(Array{Int64},zeros(qtd_itens,4))
    
    #Montando os conjuntos de classificação para usuários e itens.
    for u=1:qtd_users
        notas_existentes = find(r->r!=0,notas[u,:])
        
        for iter in notas_existentes
            if notas[u,iter] < ku
                users_sets[u,1] += 1
            elseif notas[u,iter] >= ku && notas[u,iter] < vu
                users_sets[u,2] += 1
            else
                users_sets[u,3] += 1
            end
        end        
    end
    
    for i=1:qtd_itens
        notas_existentes = find(r->r!=0,notas[:,i])
        
        for iter in notas_existentes
            if notas[iter,i] < ki
                itens_sets[i,1] += 1
                elseif notas[iter,i] >= ki && notas[iter,i] < vi
                itens_sets[i,2] += 1
            else
                itens_sets[i,3] += 1
            end
        end
        
    end
        
    #Depois de construídos, classificar cada usuário e item
    
    for iter=1:size(train)[1]
        if ( (users_sets[convert(Int64,train[iter,1]),1] < users_sets[convert(Int64,train[iter,1]),2] + 
            users_sets[convert(Int64,train[iter,1]),3]) && (itens_sets[convert(Int64,train[iter,2]),1] < 
            itens_sets[convert(Int64,train[iter,2]),2] + itens_sets[convert(Int64,train[iter,2]),3]) && 
            train[iter,3] >= k )
            push!(possiveis_ruidos,iter)
        end
        
        if ( (users_sets[convert(Int64,train[iter,1]),2] < users_sets[convert(Int64,train[iter,1]),1] + 
            users_sets[convert(Int64,train[iter,3]),3]) && (itens_sets[convert(Int64,train[iter,2]),2] < 
            itens_sets[convert(Int64,train[iter,2]),1] + 
            itens_sets[convert(Int64,train[iter,2]),3]) && (train[iter,3] < k || train[iter,3] >= v) )
            push!(possiveis_ruidos,iter)
        end
        
        if ( (users_sets[convert(Int64,train[iter,1]),3] < users_sets[convert(Int64,train[iter,1]),1] + 
            users_sets[convert(Int64,train[iter,3]),2]) && (itens_sets[convert(Int64,train[iter,2]),3] < 
            itens_sets[convert(Int64,train[iter,2]),1] + itens_sets[convert(Int64,train[iter,2]),2]) && 
            train[iter,3] < v )
            push!(possiveis_ruidos,iter)
        end
    end

    ###############################################################################################################
    
    ######################################Parte de correção do ruído###############################################
        
    for r in possiveis_ruidos
        prev = previsao(convert(Int64,train[r,1]),convert(Int64,train[r,2]),similaridades[convert(Int64,train[r,1]),:],
                        notas,60)
                
        if prev != -1            
            if abs(prev - train[r,3]) > limiar
                
                if prev < min_nota
                    prev = min_nota
                end
                
                if prev > max_nota
                    prev = max_nota
                end
                
                train[r,5] = prev
            end
        end
    end
    
    ########################################################################################################### 
end

function reset_toledo(data)
    data[:,5] = 0
end

################################################################################################################

####################################Algoritmo do KNN###############################################################
function mean_absolute_error(test,similaridades,notas,k)
    MAE = 0.0
    qtd_nao_previstos = 0
         
    for i=1:size(test)[1]
        
        prev = previsao(convert(Int64,test[i,1]),convert(Int64,test[i,2]),similaridades[convert(Int64,test[i,1]),:],notas,k)
                
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

function algoritmo_k_vizinhos_mais_proximos(train,test,k,funcao_sim)
            
    #Criação da matriz de similaridade usuários x usuários
    similaridades = eye(qtd_users,qtd_users)
    
    #########################Execução do algoritmo em si#######################################

    #Criação da matriz items x usuários
    notas = zeros(qtd_users,qtd_itens)

    for i=1:size(train)[1]
        
        if train[i,4] != -1
            notas[convert(Int64,train[i,1]),convert(Int64,train[i,2])] = train[i,3]
        else   
            notas[convert(Int64,train[i,1]),convert(Int64,train[i,2])] = -1
        end

        if train[i,5] != 0
            notas[convert(Int64,train[i,1]),convert(Int64,train[i,2])] = train[i,5]
        end
    end

    for i=2:qtd_users
        for j=1:(i-1)
            similaridades[i,j] = funcao_sim(notas[i,:], notas[j,:])
            similaridades[j,i] = similaridades[i,j]
        end
    end

    #Previsão do conjunto de teste
    aux_tupla = mean_absolute_error(test,similaridades,notas,k)
    MAE_global = aux_tupla[1]
    qtd_nao_previstos_per_fold = aux_tupla[2]
    
    ########################################################################################

    proporcao_cobertura = 100.0 - (100.0 * (qtd_nao_previstos_per_fold / size(test)[1]))
      
    return MAE_global,proporcao_cobertura
end

################################################################################################################

###########################################Implementação RSVD#############################################################
function mean_absolute_error_RSVD(test::AbstractArray,q::AbstractArray,p::AbstractArray)
    MAE = 0.0
    for i=1:size(test)[1]
        MAE += abs(test[i,3] - dot(vec(q[convert(Int64,test[i,2]),:]),vec(p[convert(Int64,test[i,1]),:])))
    end
    
    MAE /= size(test)[1]
    
    return MAE
end

#Parâmetros do método(Já determinados como ótimos pelo seu criador, Simon Funk, para o Netflix Prize)
#lambda = 0.02
#learning_rate = 0.001
function Regulared_SVD(training_set::AbstractArray,test_set::AbstractArray,lambda::Float64,learning_rate::Float64,
                       stop_criteria::Float64,q::AbstractArray,p::AbstractArray)

    error_prediction = zeros(qtd_users,qtd_itens)
    current_error::Float64 = size(data)[1]
    next_error::Float64 = 0.0

    while abs(current_error - next_error) > stop_criteria

        current_error = next_error

        for i=1:size(training_set)[1]
            
            if training_set[i,5] != 0
            
            q[convert(Int64,training_set[i,2]),:] += learning_rate * ( ( training_set[i,5] - dot(vec(q[convert(Int64,training_set[i,2]),:]),
                                      vec(p[convert(Int64,training_set[i,1]),:])) ) * p[convert(Int64,training_set[i,1]),:] - 
                                      lambda * q[convert(Int64,training_set[i,2]),:])

            p[convert(Int64,training_set[i,1]),:] += learning_rate * ( ( training_set[i,5] - dot(vec(q[convert(Int64,training_set[i,2]),:]),
                                      vec(p[convert(Int64,training_set[i,1]),:])) ) * q[convert(Int64,training_set[i,2]),:] - 
                                      lambda * p[convert(Int64,training_set[i,1]),:])                
            end
            
            if training_set[i,4] != -1
                q[convert(Int64,training_set[i,2]),:] += learning_rate * ( ( training_set[i,3] - dot(vec(q[convert(Int64,training_set[i,2]),:]),
                                      vec(p[convert(Int64,training_set[i,1]),:])) ) * p[convert(Int64,training_set[i,1]),:] - 
                                      lambda * q[convert(Int64,training_set[i,2]),:])

                p[convert(Int64,training_set[i,1]),:] += learning_rate * ( ( training_set[i,3] - dot(vec(q[convert(Int64,training_set[i,2]),:]),
                                      vec(p[convert(Int64,training_set[i,1]),:])) ) * q[convert(Int64,training_set[i,2]),:] - 
                                      lambda * p[convert(Int64,training_set[i,1]),:])
            end
            
        end
        
        next_error = 0

        for i=1:size(training_set)[1]
            
            if training_set[i,5] != 0
            
                next_error += ((training_set[i,5] - dot(vec(q[convert(Int64,training_set[i,2]),:]),
                                 vec(p[convert(Int64,training_set[i,1]),:])))^2 + lambda * (norm(q[convert(Int64,training_set[i,2]),:])^2 + 
                                 norm(p[convert(Int64,training_set[i,1]),:])^2))
            end
            
            if training_set[i,4] != -1
            
                next_error += ((training_set[i,3] - dot(vec(q[convert(Int64,training_set[i,2]),:]),
                                 vec(p[convert(Int64,training_set[i,1]),:])))^2 + lambda * (norm(q[convert(Int64,training_set[i,2]),:])^2 + 
                                 norm(p[convert(Int64,training_set[i,1]),:])^2))
            end
            
        end
    end

    #Previsão do conjunto de teste
    MAE = mean_absolute_error_RSVD(test_set,q,p)
    
    return MAE
end
################################################################################################################

#################################Inicio da construção dos testes######################################################################
#Número de "variáveis latentes"(dimensão ou fatores) que serão usados na computação da previsão.
num_of_factors = 100 

q1 = rand(qtd_itens,num_of_factors) #Item
p1 = rand(qtd_users,num_of_factors) #Usuários

q1_copy = copy(q1)
p1_copy = copy(p1)

q2 = rand(qtd_itens,num_of_factors) #Item
p2 = rand(qtd_users,num_of_factors) #Usuários

q2_copy = copy(q2)
p2_copy = copy(p2)

q3 = rand(qtd_itens,num_of_factors) #Item
p3 = rand(qtd_users,num_of_factors) #Usuários

q3_copy = copy(q3)
p3_copy = copy(p3)

q4 = rand(qtd_itens,num_of_factors) #Item
p4 = rand(qtd_users,num_of_factors) #Usuários

q4_copy = copy(q4)
p4_copy = copy(p4)

q5 = rand(qtd_itens,num_of_factors) #Item
p5 = rand(qtd_users,num_of_factors) #Usuários

q5_copy = copy(q5)
p5_copy = copy(p5)
################################################################################################################

#Rodando o KNN k = 50 e com correlação de perason

#Criação da matriz items x usuários
notas1 = zeros(qtd_users,qtd_itens)

for i=1:size(train1)[1]
    notas1[convert(Int64,train1[i,1]),convert(Int64,train1[i,2])] = train1[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades1 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades1[i,j] = correlacao_pearson(notas1[i,:], notas1[j,:])
        similaridades1[j,i] = similaridades1[i,j]
    end
end

#Criação da matriz items x usuários
notas2 = zeros(qtd_users,qtd_itens)

for i=1:size(train2)[1]
    notas2[convert(Int64,train2[i,1]),convert(Int64,train2[i,2])] = train2[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades2 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades2[i,j] = correlacao_pearson(notas2[i,:], notas2[j,:])
        similaridades2[j,i] = similaridades2[i,j]
    end
end

#Criação da matriz items x usuários
notas3 = zeros(qtd_users,qtd_itens)

for i=1:size(train3)[1]
    notas3[convert(Int64,train3[i,1]),convert(Int64,train3[i,2])] = train3[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades3 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades3[i,j] = correlacao_pearson(notas3[i,:], notas3[j,:])
        similaridades3[j,i] = similaridades3[i,j]
    end
end

#Criação da matriz items x usuários
notas4 = zeros(qtd_users,qtd_itens)

for i=1:size(train4)[1]
    notas4[convert(Int64,train4[i,1]),convert(Int64,train4[i,2])] = train4[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades4 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades4[i,j] = correlacao_pearson(notas4[i,:], notas4[j,:])
        similaridades4[j,i] = similaridades4[i,j]
    end
end

#Criação da matriz items x usuários
notas5 = zeros(qtd_users,qtd_itens)

for i=1:size(train5)[1]
    notas5[convert(Int64,train5[i,1]),convert(Int64,train5[i,2])] = train5[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades5 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades5[i,j] = correlacao_pearson(notas5[i,:], notas5[j,:])
        similaridades5[j,i] = similaridades5[i,j]
    end
end

#Rodando o KNN k = 50 e com correlação de perason

#Criação da matriz items x usuários
notas_ruido1 = zeros(qtd_users,qtd_itens)

for i=1:size(train1_ruido)[1]
    notas_ruido1[convert(Int64,train1_ruido[i,1]),convert(Int64,train1_ruido[i,2])] = train1_ruido[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades_ruido1 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades_ruido1[i,j] = correlacao_pearson(notas_ruido1[i,:], notas_ruido1[j,:])
        similaridades_ruido1[j,i] = similaridades_ruido1[i,j]
    end
end

#########################################BASES RUIDOSAS############################################################

#Criação da matriz items x usuários
notas_ruido2 = zeros(qtd_users,qtd_itens)

for i=1:size(train2_ruido)[1]
    notas_ruido2[convert(Int64,train2_ruido[i,1]),convert(Int64,train2_ruido[i,2])] = train2_ruido[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades_ruido2 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades_ruido2[i,j] = correlacao_pearson(notas_ruido2[i,:], notas_ruido2[j,:])
        similaridades_ruido2[j,i] = similaridades_ruido2[i,j]
    end
end

#Criação da matriz items x usuários
notas_ruido3 = zeros(qtd_users,qtd_itens)

for i=1:size(train3_ruido)[1]
    notas_ruido3[convert(Int64,train3_ruido[i,1]),convert(Int64,train3_ruido[i,2])] = train3_ruido[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades_ruido3 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades_ruido3[i,j] = correlacao_pearson(notas_ruido3[i,:], notas_ruido3[j,:])
        similaridades_ruido3[j,i] = similaridades_ruido3[i,j]
    end
end

#Criação da matriz items x usuários
notas_ruido4 = zeros(qtd_users,qtd_itens)

for i=1:size(train4_ruido)[1]
    notas_ruido4[convert(Int64,train4_ruido[i,1]),convert(Int64,train4_ruido[i,2])] = train4_ruido[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades_ruido4 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades_ruido4[i,j] = correlacao_pearson(notas_ruido4[i,:], notas_ruido4[j,:])
        similaridades_ruido4[j,i] = similaridades_ruido4[i,j]
    end
end

#Criação da matriz items x usuários
notas_ruido5 = zeros(qtd_users,qtd_itens)

for i=1:size(train5_ruido)[1]
    notas_ruido5[convert(Int64,train5_ruido[i,1]),convert(Int64,train5_ruido[i,2])] = train5_ruido[i,3]
end

#Criação da matriz de similaridade usuários x usuários
similaridades_ruido5 = eye(qtd_users,qtd_users)    

for i=2:qtd_users
    for j=1:(i-1)
        similaridades_ruido5[i,j] = correlacao_pearson(notas_ruido5[i,:], notas_ruido5[j,:])
        similaridades_ruido5[j,i] = similaridades_ruido5[i,j]
    end
end
################################################################################################################

##########################################TODOS OS TESTES##########################################################

######################################KNN################################################

#KNN com geração de ruído na base

#println("KNN com geração de ruído na base: 20%")

#println(algoritmo_k_vizinhos_mais_proximos(train1_ruido,test1,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train2_ruido,test2,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train3_ruido,test3,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train4_ruido,test4,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train5_ruido,test5,50,correlacao_pearson))

#KNN com geração de ruído na base - Mahony

#println("KNN com geração de ruído na base - Mahony:")

#correcao_ruido_mahony(train1_ruido,0.55,1,5,notas_ruido1,similaridades_ruido1)
#correcao_ruido_mahony(train2_ruido,0.55,1,5,notas_ruido2,similaridades_ruido2)
#correcao_ruido_mahony(train3_ruido,0.55,1,5,notas_ruido3,similaridades_ruido3)
#correcao_ruido_mahony(train4_ruido,0.55,1,5,notas_ruido4,similaridades_ruido4)
#correcao_ruido_mahony(train5_ruido,0.55,1,5,notas_ruido5,similaridades_ruido5)

#println(algoritmo_k_vizinhos_mais_proximos(train1_ruido,test1,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train2_ruido,test2,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train3_ruido,test3,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train4_ruido,test4,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train5_ruido,test5,50,correlacao_pearson))

#reset_mahony(train1_ruido)
#reset_mahony(train2_ruido)
#reset_mahony(train3_ruido)
#reset_mahony(train4_ruido)
#reset_mahony(train5_ruido)

#KNN com geração de ruído na base - Toledo

#println("KNN com geração de ruído na base - Toledo")

#correcao_ruido_toledo(train1_ruido,1,5,collect(1:5),notas_ruido1,similaridades_ruido1)
#correcao_ruido_toledo(train2_ruido,1,5,collect(1:5),notas_ruido2,similaridades_ruido2)
#correcao_ruido_toledo(train3_ruido,1,5,collect(1:5),notas_ruido3,similaridades_ruido3)
#correcao_ruido_toledo(train4_ruido,1,5,collect(1:5),notas_ruido4,similaridades_ruido4)
#correcao_ruido_toledo(train5_ruido,1,5,collect(1:5),notas_ruido5,similaridades_ruido5)

#println(algoritmo_k_vizinhos_mais_proximos(train1_ruido,test1,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train2_ruido,test2,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train3_ruido,test3,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train4_ruido,test4,50,correlacao_pearson))
#println(algoritmo_k_vizinhos_mais_proximos(train5_ruido,test5,50,correlacao_pearson))

#reset_toledo(train1_ruido)
#reset_toledo(train2_ruido)
#reset_toledo(train3_ruido)
#reset_toledo(train4_ruido)
#reset_toledo(train5_ruido)


##########################################################################################


###################################RSVD####################################################

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#RSVD

#println("RSVD:")

#println(Regulared_SVD(train1,test1,0.02,0.001,1.0,q1,p1))
#println(Regulared_SVD(train2,test2,0.02,0.001,1.0,q2,p2))
#println(Regulared_SVD(train3,test3,0.02,0.001,1.0,q3,p3))
#println(Regulared_SVD(train4,test4,0.02,0.001,1.0,q4,p4))
#println(Regulared_SVD(train5,test5,0.02,0.001,1.0,q5,p5))

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#RSVD com Mahony: (sem geração de ruído natural)

#println("RSVD com Mahony: (sem geração de ruído natural):")

#correcao_ruido_mahony(train1,0.55,1,5,notas1,similaridades1)
#correcao_ruido_mahony(train2,0.55,1,5,notas2,similaridades2)
#correcao_ruido_mahony(train3,0.55,1,5,notas3,similaridades3)
#correcao_ruido_mahony(train4,0.55,1,5,notas4,similaridades4)
#correcao_ruido_mahony(train5,0.55,1,5,notas5,similaridades5)

#println(Regulared_SVD(train1,test1,0.02,0.001,1.0,q1,p1))
#println(Regulared_SVD(train2,test2,0.02,0.001,1.0,q2,p2))
#println(Regulared_SVD(train3,test3,0.02,0.001,1.0,q3,p3))
#println(Regulared_SVD(train4,test4,0.02,0.001,1.0,q4,p4))
#println(Regulared_SVD(train5,test5,0.02,0.001,1.0,q5,p5))

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#reset_mahony(train1)
#reset_mahony(train2)
#reset_mahony(train3)
#reset_mahony(train4)
#reset_mahony(train5)

#RSVD com Toledo: (sem geração de ruído natural)

#println("RSVD com Toledo: (sem geração de ruído natural):")

#correcao_ruido_toledo(train1,1,5,collect(1:5),notas1,similaridades1)
#correcao_ruido_toledo(train2,1,5,collect(1:5),notas2,similaridades2)
#correcao_ruido_toledo(train3,1,5,collect(1:5),notas3,similaridades3)
#correcao_ruido_toledo(train4,1,5,collect(1:5),notas4,similaridades4)
#correcao_ruido_toledo(train5,1,5,collect(1:5),notas5,similaridades5)

#println(Regulared_SVD(train1,test1,0.02,0.001,1.0,q1,p1))
#println(Regulared_SVD(train2,test2,0.02,0.001,1.0,q2,p2))
#println(Regulared_SVD(train3,test3,0.02,0.001,1.0,q3,p3))
#println(Regulared_SVD(train4,test4,0.02,0.001,1.0,q4,p4))
#println(Regulared_SVD(train5,test5,0.02,0.001,1.0,q5,p5))

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#reset_toledo(train1)
#reset_toledo(train2)
#reset_toledo(train3)
#reset_toledo(train4)
#reset_toledo(train5)

#RSVD com geração de ruído

#println("RSVD com geração de ruído: 20%")

#println(Regulared_SVD(train1_ruido,test1,0.02,0.001,1.0,q1,p1))
#println(Regulared_SVD(train2_ruido,test2,0.02,0.001,1.0,q2,p2))
#println(Regulared_SVD(train3_ruido,test3,0.02,0.001,1.0,q3,p3))
#println(Regulared_SVD(train4_ruido,test4,0.02,0.001,1.0,q4,p4))
#println(Regulared_SVD(train5_ruido,test5,0.02,0.001,1.0,q5,p5))

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#RSVD com geração de ruído - Mahony

#println("RSVD com geração de ruído - Mahony:")

#correcao_ruido_mahony(train1_ruido,0.55,1,5,notas_ruido1,similaridades_ruido1)
#correcao_ruido_mahony(train2_ruido,0.55,1,5,notas_ruido2,similaridades_ruido2)
#correcao_ruido_mahony(train3_ruido,0.55,1,5,notas_ruido3,similaridades_ruido3)
#correcao_ruido_mahony(train4_ruido,0.55,1,5,notas_ruido4,similaridades_ruido4)
#correcao_ruido_mahony(train5_ruido,0.55,1,5,notas_ruido5,similaridades_ruido5)

#println(Regulared_SVD(train1_ruido,test1,0.02,0.001,1.0,q1,p1))
#println(Regulared_SVD(train2_ruido,test2,0.02,0.001,1.0,q2,p2))
#println(Regulared_SVD(train3_ruido,test3,0.02,0.001,1.0,q3,p3))
#println(Regulared_SVD(train4_ruido,test4,0.02,0.001,1.0,q4,p4))
#println(Regulared_SVD(train5_ruido,test5,0.02,0.001,1.0,q5,p5))

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#reset_mahony(train1_ruido)
#reset_mahony(train2_ruido)
#reset_mahony(train3_ruido)
#reset_mahony(train4_ruido)
#reset_mahony(train5_ruido)

#RSVD com geração de ruído - Toledo

#println("RSVD com geração de ruído - Toledo:")

#correcao_ruido_toledo(train1_ruido,1,5,collect(1:5),notas_ruido1,similaridades_ruido1)
#correcao_ruido_toledo(train2_ruido,1,5,collect(1:5),notas_ruido2,similaridades_ruido2)
#correcao_ruido_toledo(train3_ruido,1,5,collect(1:5),notas_ruido3,similaridades_ruido3)
#correcao_ruido_toledo(train4_ruido,1,5,collect(1:5),notas_ruido4,similaridades_ruido4)
#correcao_ruido_toledo(train5_ruido,1,5,collect(1:5),notas_ruido5,similaridades_ruido5)

#println(Regulared_SVD(train1_ruido,test1,0.02,0.001,1.0,q1,p1))
#println(Regulared_SVD(train2_ruido,test2,0.02,0.001,1.0,q2,p2))
#println(Regulared_SVD(train3_ruido,test3,0.02,0.001,1.0,q3,p3))
#println(Regulared_SVD(train4_ruido,test4,0.02,0.001,1.0,q4,p4))
#println(Regulared_SVD(train5_ruido,test5,0.02,0.001,1.0,q5,p5))

#q1 = copy(q1_copy)
#p1 = copy(p1_copy)
#q2 = copy(q2_copy)
#p2 = copy(p2_copy)
#q3 = copy(q3_copy)
#p3 = copy(p4_copy)
#q4 = copy(q4_copy)
#p4 = copy(p4_copy)
#q5 = copy(q5_copy)
#p5 = copy(p5_copy)

#reset_toledo(train1_ruido)
#reset_toledo(train2_ruido)
#reset_toledo(train3_ruido)
#reset_toledo(train4_ruido)
#reset_toledo(train5_ruido)
