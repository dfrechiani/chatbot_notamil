import streamlit as st
import os
import time
import re
import json
from datetime import datetime
from openai import OpenAIError
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Constantes
MAX_TOKENS = 4000
TEMPERATURA = 0.2
OPENAI_API_KEY = st.secrets["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"]
MODELO_FINETUNED = "ft:gpt-4o-2024-08-06:personal::A3uFSo9x"

@st.cache_resource
def carregar_faiss():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        base_conhecimento = FAISS.load_local("/Users/danielfrechiani/Desktop/APP.REDACAO", embeddings, allow_dangerous_deserialization=True)
        return base_conhecimento
    except Exception as e:
        st.error(f"Erro ao carregar o índice FAISS: {e}")
        return None

@st.cache_resource
def inicializar_rag():
    base_conhecimento = carregar_faiss()
    if base_conhecimento is None:
        st.error("Erro ao carregar a base de conhecimento FAISS.")
        return None
    try:
        llm = ChatOpenAI(model=MODELO_FINETUNED, openai_api_key=OPENAI_API_KEY, temperature=TEMPERATURA, max_tokens=MAX_TOKENS)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=base_conhecimento.as_retriever(),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Erro ao inicializar o RAG: {e}")
        return None

def processar_rag_com_pausa(qa_chain, query: str, tentativas_max=5):
    for tentativa in range(tentativas_max):
        try:
            resposta = qa_chain.invoke({"query": query})
            resultado = resposta.get('result', '')
            return resultado
        except Exception as e:
            st.warning(f"Erro na tentativa {tentativa + 1}: {e}")
            if tentativa < tentativas_max - 1:
                time.sleep(60)
            else:
                st.error("Número máximo de tentativas atingido. Tente novamente mais tarde.")
    return None

def extrair_erros(feedback):
    erros = re.findall(r"([\w\s]+):\s*(.*?)(?=\w+:|$)", feedback, re.DOTALL)
    return [(categoria.strip(), descricao.strip()) for categoria, descricao in erros]

def processar_redacao_completa(qa_chain, redacao_texto: str, tema_redacao: str):
    prompt_completo = f"""
    Atue como um tutor especializado em língua portuguesa e redação para ENEM, utilizando linguagem dialógica e amigável. Você é um avaliador de redação e fornece feedback qualificado para que o usuário consiga obter nota mil na redação do ENEM.

    Analise a seguinte redação sobre o tema "{tema_redacao}". Forneça uma análise completa seguindo estritamente o formato abaixo:

    [INÍCIO DA ANÁLISE]

    1. Domínio da Norma Culta (0-200 pontos):
    [Nota numérica]/200
    [Feedback específico sobre gramática, ortografia e pontuação, com exemplos de correções]

    2. Compreensão do Tema (0-200 pontos):
    [Nota numérica]/200
    [Feedback sobre a relevância e profundidade da abordagem do tema, sugerindo inclusões para enriquecimento]

    3. Seleção e Organização das Informações (0-200 pontos):
    [Nota numérica]/200
    [Feedback sobre a lógica e clareza da argumentação, com sugestões para melhorar a estrutura]

    4. Conhecimento dos Mecanismos Linguísticos (0-200 pontos):
    [Nota numérica]/200
    [Feedback sobre o uso de recursos linguísticos para argumentação, com exemplos para aprimoramento]

    5. Proposta de Intervenção (0-200 pontos):
    [Nota numérica]/200
    [Feedback sobre a proposta de solução, incentivando detalhamento e viabilidade]

    Nota Total: [Soma das notas acima]/1000

    Comentário Geral:
    [Forneça um comentário geral de 2-3 frases sobre os principais pontos fortes e fracos da redação]

    Competências que precisam de mais atenção:
    [Liste as competências com menor pontuação]

    [FIM DA ANÁLISE]

    Convite para Trilha de Aprendizado:
    [Faça uma pergunta clara sobre o interesse do aluno em prosseguir em uma trilha de aprendizado personalizada para reforçar seus pontos de melhoria]

    Redação a ser analisada:
    {redacao_texto}
    """
    
    resultado_completo = processar_rag_com_pausa(qa_chain, prompt_completo)
    
    if resultado_completo is None:
        return None

    analise_processada = {}
    competencias = [
        "Domínio da Norma Culta",
        "Compreensão do Tema",
        "Seleção e Organização das Informações",
        "Conhecimento dos Mecanismos Linguísticos",
        "Proposta de Intervenção"
    ]

    for i, competencia in enumerate(competencias, 1):
        padrao = rf"{i}\. {competencia} \(0-200 pontos\):\s*(\d+)/200\s*(.*?)(?=\d+\. |\[FIM DA ANÁLISE\])"
        match = re.search(padrao, resultado_completo, re.DOTALL)
        if match:
            nota = int(match.group(1))
            feedback = match.group(2).strip()
            erros = extrair_erros(feedback)
            analise_processada[competencia] = {
                "nota": nota,
                "feedback": feedback,
                "erros": erros
            }

    nota_total_match = re.search(r"Nota Total: (\d+)/1000", resultado_completo)
    nota_total = int(nota_total_match.group(1)) if nota_total_match else 0

    comentario_geral_match = re.search(r"Comentário Geral:(.*?)Competências que precisam de mais atenção:", resultado_completo, re.DOTALL)
    comentario_geral = comentario_geral_match.group(1).strip() if comentario_geral_match else ""

    competencias_atencao_match = re.search(r"Competências que precisam de mais atenção:(.*?)\[FIM DA ANÁLISE\]", resultado_completo, re.DOTALL)
    competencias_atencao = competencias_atencao_match.group(1).strip() if competencias_atencao_match else ""

    convite_trilha_match = re.search(r"Convite para Trilha de Aprendizado:(.*?)$", resultado_completo, re.DOTALL)
    convite_trilha = convite_trilha_match.group(1).strip() if convite_trilha_match else ""

    return {
        "analise_processada": analise_processada,
        "nota_total": nota_total,
        "comentario_geral": comentario_geral,
        "competencias_atencao": competencias_atencao,
        "convite_trilha": convite_trilha,
        "redacao_texto": redacao_texto
    }

def trilha_correcao(qa_chain, analise_processada, redacao_texto):
    st.subheader("Trilha de Correção Personalizada")
    
    competencias = list(analise_processada.keys())
    competencia_escolhida = st.selectbox("Escolha a competência que deseja trabalhar:", competencias)
    
    if st.button("Iniciar Trabalho na Competência"):
        dados = analise_processada[competencia_escolhida]
        st.write(f"Trabalhando na {competencia_escolhida}")
        st.write(f"Nota atual: {dados['nota']}/200")

        for erro in dados['erros']:
            categoria, descricao = erro
            st.write(f"\nErro identificado: {categoria} - {descricao}")

            prompt_teoria = f"""
            Busque na base de conhecimento e apresente:
            1. Uma explicação simples e didática sobre a teoria relacionada ao erro '{categoria}' na competência {competencia_escolhida}.
            2. A importância dessa regra ou conceito na escrita.
            3. Se possível, um exemplo correto de uso.
            """
            teoria = processar_rag_com_pausa(qa_chain, prompt_teoria)
            st.write("Teoria:")
            st.write(teoria)

            prompt_exercicios = f"""
            Com base na teoria apresentada, forneça:
            1. Dois exercícios simples para praticar a correção do erro '{categoria}' na competência {competencia_escolhida}.
            2. As respostas corretas para cada exercício.

            Formato da resposta:
            Exercícios:
            1. [Primeiro exercício]
            2. [Segundo exercício]

            Respostas:
            1. [Resposta do primeiro exercício]
            2. [Resposta do segundo exercício]
            """
            exercicios_e_respostas = processar_rag_com_pausa(qa_chain, prompt_exercicios)
            
            try:
                exercicios, respostas = exercicios_e_respostas.split("Respostas:", 1)
            except ValueError:
                exercicios = exercicios_e_respostas
                respostas = "Não foi possível gerar respostas separadamente."
            
            st.write("Exercícios preparatórios:")
            st.write(exercicios.strip())
            
            resposta_aluno1 = st.text_input("Sua resposta para o exercício 1:")
            resposta_aluno2 = st.text_input("Sua resposta para o exercício 2:")
            
            if st.button("Verificar Respostas"):
                st.write("Respostas corretas:")
                st.write(respostas.strip())

            st.write("Agora, aplique o que aprendeu para corrigir o erro no texto original.")
            st.write(f"Texto original: {redacao_texto}")
            correcao_aluno = st.text_area("Digite sua correção:")

            if st.button("Avaliar Correção"):
                prompt_avaliacao = f"""
                Avalie a correção do aluno para o erro '{categoria}' na competência {competencia_escolhida}.
                Erro original: {descricao}
                Correção do aluno: {correcao_aluno}
                Forneça:
                1. Um feedback construtivo sobre a correção.
                2. Se a correção está totalmente correta, parcialmente correta ou incorreta.
                3. Sugestões de melhoria, se necessário.
                4. A forma correta de correção, se o aluno não acertou completamente.
                """
                avaliacao = processar_rag_com_pausa(qa_chain, prompt_avaliacao)
                st.write("Avaliação da sua correção:")
                st.write(avaliacao)

        st.success(f"Você completou todos os erros da {competencia_escolhida}!")

def main():
    st.title("Corretor de Redação Interativo com Trilha Personalizada")

    nome_aluno = st.text_input("Nome do Aluno:")
    tema_redacao = st.text_input("Tema da Redação:")
    redacao_texto = st.text_area("Cole sua redação aqui:", height=300)

    if st.button("Corrigir Redação"):
        if nome_aluno and tema_redacao and redacao_texto:
            with st.spinner('Processando a redação...'):
                qa_chain = inicializar_rag()
                if qa_chain:
                    resultados = processar_redacao_completa(qa_chain, redacao_texto, tema_redacao)
                    if resultados:
                        st.subheader(f"Análise da redação de {nome_aluno}")
                        st.write(f"Nota Total: {resultados['nota_total']}/1000")
                        st.write("Comentário Geral:")
                        st.write(resultados['comentario_geral'])
                        st.write("Competências que precisam de mais atenção:")
                        st.write(resultados['competencias_atencao'])
                        st.write("Convite para Trilha de Aprendizado:")
                        st.write(resultados['convite_trilha'])
                        
                        for competencia, dados in resultados['analise_processada'].items():
                            st.write(f"\n{competencia}:")
                            st.write(f"Nota: {dados['nota']}/200")
                            st.write(f"Feedback: {dados['feedback']}")
                        
                        if st.button("Iniciar Trilha de Correção"):
                            trilha_correcao(qa_chain, resultados['analise_processada'], redacao_texto)
                    else:
                        st.error("Não foi possível processar a análise da redação.")
                else:
                    st.error("Não foi possível inicializar o sistema de correção.")
        else:
            st.warning("Por favor, preencha todos os campos.")

if __name__ == "__main__":
    main()
