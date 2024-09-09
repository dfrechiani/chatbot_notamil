import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import time
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="Análise de Redação ENEM",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"),
   

# Carregando o CSS personalizado
def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Configurações
OPENAI_API_KEY = st.secrets["openai_api_key"]  # Obtendo a chave da API dos secrets
MAX_TOKENS = 4000
TEMPERATURA = 0.2
MODELO_FINETUNED = "ft:gpt-4o-2024-08-06:personal::A3uFSo9x"

# Função para carregar o índice FAISS
@st.cache_resource
def carregar_faiss():
    PASTA_SAIDA = "/Users/danielfrechiani/Desktop/APP.REDACAO/"
    INDICE_FAISS = os.path.join(PASTA_SAIDA, "novo_index.faiss")
    
    if not os.path.exists(INDICE_FAISS):
        st.error(f"Arquivo {INDICE_FAISS} não encontrado.")
        return None
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    try:
        base_conhecimento = FAISS.load_local(INDICE_FAISS, embeddings, allow_dangerous_deserialization=True)
        return base_conhecimento
    except Exception as e:
        st.error(f"Erro ao carregar o índice FAISS: {e}")
        return None

# Função para inicializar o RAG
@st.cache_resource
def inicializar_rag():
    base_conhecimento = carregar_faiss()
    if base_conhecimento is None:
        return None
    
    try:
        llm = ChatOpenAI(
            model=MODELO_FINETUNED,
            openai_api_key=OPENAI_API_KEY,
            temperature=TEMPERATURA,
            max_tokens=MAX_TOKENS
        )
        return llm
    except Exception as e:
        st.error(f"Erro ao inicializar o RAG: {e}")
        return None

# Função para inicializar o RAG
@st.cache_resource
def inicializar_rag():
    base_conhecimento = carregar_faiss()
    if base_conhecimento is None:
        return None
    
    try:
        llm = ChatOpenAI(
            model=MODELO_FINETUNED,
            openai_api_key=OPENAI_API_KEY,
            temperature=TEMPERATURA,
            max_tokens=MAX_TOKENS
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=base_conhecimento.as_retriever(),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Erro ao inicializar o RAG: {e}")
        return None

def processar_rag_com_pausa(qa_chain, query: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = qa_chain.invoke({"query": query})
            return response['result']
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Tentativa {attempt + 1} falhou. Tentando novamente...")
                time.sleep(2)
            else:
                st.error(f"Falha após {max_retries} tentativas. Erro: {str(e)}")
                return None

def analisar_redacao_completa(qa_chain, redacao_texto: str, tema_redacao: str):
    competencias = [
        "Domínio da Norma Culta",
        "Compreensão do Tema",
        "Seleção e Organização das Informações",
        "Conhecimento dos Mecanismos Linguísticos",
        "Proposta de Intervenção"
    ]
    
    prompt_completo = f"""
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

    Importante: As notas para cada competência devem ser múltiplos de 40 (0, 40, 80, 120, 160 ou 200).

    Redação a ser analisada:
    {redacao_texto}
    """
    
    resultado_completo = processar_rag_com_pausa(qa_chain, prompt_completo)
    
    if resultado_completo is None:
        return None

    analise_processada = ""
    nota_total = 0
    for i, comp in enumerate(competencias, 1):
        padrao = rf"{i}\. {comp} \(0-200 pontos\):\s*(\d+)/200\s*(.*?)(?=\d+\. |\[FIM DA ANÁLISE\])"
        match = re.search(padrao, resultado_completo, re.DOTALL)
        if match:
            nota_bruta = int(match.group(1))
            nota = min(round(nota_bruta / 40) * 40, 200)
            feedback = match.group(2).strip()
            analise_processada += f"{comp}:\nNota: {nota}/200\nFeedback: {feedback}\n\n"
            nota_total += nota

    comentario_geral_match = re.search(r"Comentário Geral:(.*?)Competências que precisam de mais atenção:", resultado_completo, re.DOTALL)
    comentario_geral = comentario_geral_match.group(1).strip() if comentario_geral_match else ""

    competencias_atencao_match = re.search(r"Competências que precisam de mais atenção:(.*?)\[FIM DA ANÁLISE\]", resultado_completo, re.DOTALL)
    competencias_atencao = competencias_atencao_match.group(1).strip() if competencias_atencao_match else ""

    analise_final = f"""
Análise da Redação:

{analise_processada}
Nota Total: {nota_total}/1000

Comentário Geral:
{comentario_geral}

Competências que precisam de mais atenção:
{competencias_atencao}

Redação analisada:
{redacao_texto}
"""

    return analise_final.strip()


def trilha_correcao_personalizada(qa_chain, competencia: str, redacao_texto: str, feedback_original: str):
    st.markdown(f"<h2 class='sub-title'>Trilha de Correção Personalizada: {competencia}</h2>", unsafe_allow_html=True)
    
    if 'etapa_trilha' not in st.session_state:
        st.session_state['etapa_trilha'] = 'apresentacao'
        st.session_state['nivel_atual'] = 'Básico'
        st.session_state['erros_identificados'] = None
        st.session_state['categorias_erros'] = None
        st.session_state['categoria_atual'] = 0

    if st.session_state['etapa_trilha'] == 'apresentacao':
        prompt_apresentacao = f"""
        Apresente a Competência '{competencia}' do ENEM, explicando sua importância 
        e o que será avaliado. Forneça uma visão geral amigável para contextualizar o estudante.
        Use informações da base de conhecimento gramatical se disponível.
        """
        apresentacao = processar_rag_com_pausa(qa_chain, prompt_apresentacao)
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### Visão Geral da Competência")
        st.write(apresentacao)
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state['etapa_trilha'] = 'identificacao_erros'
        st.rerun()

    elif st.session_state['etapa_trilha'] == 'identificacao_erros':
        if st.session_state['erros_identificados'] is None:
            prompt_identificacao = f"""
            Analise o seguinte texto, identificando todos os erros possíveis 
            relacionados à Competência '{competencia}' do ENEM. 
            Liste cada erro encontrado, indicando o tipo de erro.
            Use a base de conhecimento gramatical para identificar os erros com precisão.

            Texto da redação:
            {redacao_texto}
            """
            erros_identificados = processar_rag_com_pausa(qa_chain, prompt_identificacao)
            
            prompt_agrupamento = f"""
            Agrupe os erros identificados por categoria, considerando a Competência '{competencia}':

            {erros_identificados}

            Apresente os resultados em forma de lista, onde cada categoria é seguida por ':' e seus erros correspondentes.
            Se não houver categorias claras, liste os erros individualmente.
            """
            erros_agrupados = processar_rag_com_pausa(qa_chain, prompt_agrupamento)
            
            st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
            st.markdown("### Erros Identificados e Agrupados")
            st.write(erros_agrupados)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state['erros_identificados'] = erros_agrupados
            st.session_state['categorias_erros'] = extrair_categorias(erros_agrupados)
        
        st.session_state['etapa_trilha'] = 'correcao_erros'
        st.rerun()

    elif st.session_state['etapa_trilha'] == 'correcao_erros':
        if st.session_state['categoria_atual'] < len(st.session_state['categorias_erros']):
            categoria = st.session_state['categorias_erros'][st.session_state['categoria_atual']]
            st.markdown(f"<div class='trilha-step'>", unsafe_allow_html=True)
    elif st.session_state['etapa_trilha'] == 'correcao_erros':
        if st.session_state['categoria_atual'] < len(st.session_state['categorias_erros']):
            categoria = st.session_state['categorias_erros'][st.session_state['categoria_atual']]
            st.markdown(f"<div class='trilha-step'>", unsafe_allow_html=True)
            st.markdown(f"<h3 class='sub-title'>Trabalhando no erro: {categoria}</h3>", unsafe_allow_html=True)

            prompt_teoria = f"""
            Explique a teoria ou conceito relacionado ao erro '{categoria}' na Competência '{competencia}'.
            Forneça uma explicação clara e concisa, adequada para um estudante do Ensino Médio.
            Use exemplos e explicações da base de conhecimento gramatical para enriquecer a resposta.
            Se não houver informações específicas na base de conhecimento, forneça uma explicação geral baseada em princípios gramaticais comuns.
            """
            teoria = processar_rag_com_pausa(qa_chain, prompt_teoria)
            st.markdown("<div class='trilha-teoria'>", unsafe_allow_html=True)
            st.markdown("### Teoria")
            st.write(teoria)
            st.markdown("</div>", unsafe_allow_html=True)

            prompt_exercicio = f"""
            Crie um exercício de nível {st.session_state['nivel_atual']} para praticar a correção do erro '{categoria}'
            na Competência '{competencia}'. O exercício deve ser focado em ajudar o aluno a compreender 
            e abordar o tema de forma mais precisa e relevante.
            Use exemplos ou contextos da base de conhecimento gramatical para criar um exercício relevante e preciso.
            Forneça também a resposta correta para este exercício.

            Formato da resposta:
            Exercício: [Texto do exercício]
            Resposta correta: [Resposta correta do exercício]
            """
            resultado_exercicio = processar_rag_com_pausa(qa_chain, prompt_exercicio)
            
            exercicio_match = re.search(r'Exercício:(.*?)Resposta correta:', resultado_exercicio, re.DOTALL)
            resposta_correta_match = re.search(r'Resposta correta:(.*)', resultado_exercicio, re.DOTALL)
            
            if exercicio_match and resposta_correta_match:
                exercicio = exercicio_match.group(1).strip()
                resposta_correta = resposta_correta_match.group(1).strip()

                st.markdown("<div class='trilha-exercicio'>", unsafe_allow_html=True)
                st.markdown(f"### Exercício de Nível {st.session_state['nivel_atual']}")
                st.write(exercicio)
                st.markdown("</div>", unsafe_allow_html=True)
                
                resposta_aluno = st.text_area(f"Sua resposta ao exercício:", key=f"resposta_{st.session_state['categoria_atual']}_{st.session_state['nivel_atual']}")
                
                if st.button("Verificar resposta"):
                    prompt_avaliacao = f"""
                    Avalie a seguinte resposta do aluno ao exercício sobre '{categoria}' na Competência '{competencia}':

                    Exercício: {exercicio}
                    Resposta correta: {resposta_correta}
                    Resposta do aluno: {resposta_aluno}

                    Determine se a resposta do aluno está correta. Se estiver incorreta, forneça um feedback detalhado,
                    indicando o que está errado e como pode ser melhorado. Use a base de conhecimento gramatical
                    para fundamentar a explicação.

                    Responda no formato:
                    Correto: [Sim/Não]
                    Feedback: [Seu feedback detalhado]
                    """
                    avaliacao = processar_rag_com_pausa(qa_chain, prompt_avaliacao)
                    
                    correto_match = re.search(r'Correto:\s*(Sim|Não)', avaliacao, re.IGNORECASE)
                    feedback_match = re.search(r'Feedback:(.*)', avaliacao, re.DOTALL)
                    
                    if correto_match and feedback_match:
                        exercicio_correto = correto_match.group(1).lower() == 'sim'
                        feedback = feedback_match.group(1).strip()
                        
                        st.markdown("<div class='trilha-feedback'>", unsafe_allow_html=True)
                        st.markdown("### Avaliação da Resposta")
                        st.write(f"**Resultado:** {'Correto' if exercicio_correto else 'Incorreto'}")
                        st.write(f"**Feedback:** {feedback}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if exercicio_correto:
                            st.success(f"Parabéns! Você completou o exercício de nível {st.session_state['nivel_atual']}.")
                            if st.session_state['nivel_atual'] != "Avançado":
                                if st.button("Continuar para o próximo nível"):
                                    if st.session_state['nivel_atual'] == 'Básico':
                                        st.session_state['nivel_atual'] = 'Intermediário'
                                    elif st.session_state['nivel_atual'] == 'Intermediário':
                                        st.session_state['nivel_atual'] = 'Avançado'
                                    st.experimental_rerun()
                                if st.button("Passar para o próximo erro"):
                                    st.session_state['categoria_atual'] += 1
                                    st.session_state['nivel_atual'] = 'Básico'
                                    st.experimental_rerun()
                            else:
                                st.success("Você completou todos os níveis para esta categoria de erro!")
                                st.session_state['categoria_atual'] += 1
                                st.session_state['nivel_atual'] = 'Básico'
                                st.experimental_rerun()
                        else:
                            st.warning("Você não acertou completamente. Tente novamente ou peça ajuda adicional.")
                    else:
                        st.error("Não foi possível avaliar a resposta. Por favor, tente novamente.")
            else:
                st.error(f"Não foi possível gerar o exercício. Passando para o próximo erro.")
                st.session_state['categoria_atual'] += 1
                st.experimental_rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success("Parabéns! Você completou todos os exercícios para esta competência.")
            st.session_state['etapa_trilha'] = 'analise_final'
            st.experimental_rerun()

    elif st.session_state['etapa_trilha'] == 'analise_final':
        nivel_proficiencia = st.session_state['nivel_atual']
        prompt_analise = f"""
        Com base nas correções realizadas para a Competência '{competencia}' do ENEM, crie:

        1. Um mapa de fragilidades, destacando as áreas onde o aluno apresentou maior dificuldade.
        2. Um breve relatório estatístico, apontando as áreas de maior defasagem e seu impacto potencial na nota do ENEM.
        3. Sugestões específicas para estudo futuro focado nesta competência.
        4. Recursos e estratégias de estudo recomendados.

        Use as informações das correções, o nível de proficiência final do aluno ({nivel_proficiencia}),
        e a base de conhecimento gramatical para fornecer recomendações precisas e relevantes.
        """
        analise_final = processar_rag_com_pausa(qa_chain, prompt_analise)
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("## Análise Final e Próximos Passos")
        st.write(analise_final)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Concluir Trilha de Correção"):
            st.session_state['etapa_trilha'] = 'apresentacao'  # Reset para próxima competência
            return nivel_proficiencia, analise_final

    st.markdown(f"<div class='trilha-progress'>Progresso: {st.session_state['categoria_atual'] + 1}/{len(st.session_state['categorias_erros'])} erros</div>", unsafe_allow_html=True)
    st.rerun()

def extrair_categorias(erros_agrupados):
    categorias = re.findall(r'^([A-Z][^:]+):', erros_agrupados, re.MULTILINE)
    if not categorias:
        categorias = [linha.strip() for linha in erros_agrupados.split('\n') if linha.strip()]
    return list(set(categorias))

def main():
        # Sidebar
    with st.sidebar:
        st.image("logo.png", width=200)  # Substitua com o caminho para o seu logo
        st.markdown("<h1 class='main-title'>Análise de Redação ENEM</h1>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Este aplicativo analisa redações do ENEM, fornece feedback detalhado e oferece uma trilha de correção personalizada.</div>", unsafe_allow_html=True)

    # Main content
    if 'page' not in st.session_state:
        st.session_state['page'] = 'input'

    if st.session_state['page'] == 'input':
        display_input_page()
    elif st.session_state['page'] == 'analysis':
        display_analysis_page()
    elif st.session_state['page'] == 'trilha':
        display_trilha_page()

def display_input_page():
    st.markdown("<h1 class='main-title'>Análise de Redação ENEM</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        nome_aluno = st.text_input("Nome do Aluno:")
        tema_redacao = st.text_input("Tema da Redação:")
        redacao_texto = st.text_area("Cole sua redação aqui:", height=300)

    with col2:
        st.markdown("""
        <div class='info-box'>
        <h2>Como funciona?</h2>
        <ol>
            <li>Insira seus dados</li>
            <li>Cole sua redação</li>
            <li>Clique em 'Analisar Redação'</li>
            <li>Receba feedback detalhado</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Analisar Redação", key="analisar_button"):
        if not nome_aluno or not tema_redacao or not redacao_texto:
            st.markdown("<div class='error-box'>Por favor, preencha todos os campos.</div>", unsafe_allow_html=True)
        else:
            with st.spinner('Analisando sua redação...'):
                qa_chain = inicializar_rag()
                if qa_chain:
                    resultado = analisar_redacao_completa(qa_chain, redacao_texto, tema_redacao)
                    if resultado:
                        st.session_state['resultado'] = resultado
                        st.session_state['nome_aluno'] = nome_aluno
                        st.session_state['qa_chain'] = qa_chain
                        st.session_state['page'] = 'analysis'
                        st.rerun()
                    else:
                        st.markdown("<div class='error-box'>Não foi possível processar a análise da redação.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-box'>Não foi possível inicializar o sistema de análise.</div>", unsafe_allow_html=True)

def display_analysis_page():
    resultado = st.session_state['resultado']
    nome_aluno = st.session_state['nome_aluno']

    st.markdown(f"<h1 class='main-title'>Resultado da Análise - {nome_aluno}</h1>", unsafe_allow_html=True)

    # Dividir o resultado em linhas
    linhas = resultado.split('\n')

    # Extrair nota total
    nota_total = next((linha.split(':')[1].strip() for linha in linhas if linha.startswith('Nota Total:')), 'N/A')

    # Extrair notas por competência
    competencias = ['Domínio da Norma Culta', 'Compreensão do Tema', 'Seleção e Organização das Informações', 'Conhecimento dos Mecanismos Linguísticos', 'Proposta de Intervenção']
    notas = []
    for comp in competencias:
        nota = next((linha.split(':')[1].strip().split('/')[0] for linha in linhas if linha.startswith(f"{comp}:\nNota:")), '0')
        notas.append(int(nota))

    # Nota Total e métricas principais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Nota Total", nota_total)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Competência Mais Forte", competencias[notas.index(max(notas))])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Competência a Melhorar", competencias[notas.index(min(notas))])
        st.markdown("</div>", unsafe_allow_html=True)

    # Gráfico de notas por competência
    st.markdown("<h2 class='sub-title'>Notas por Competência</h2>", unsafe_allow_html=True)
    fig = px.bar(
        x=competencias,
        y=notas,
        labels={'x': 'Competência', 'y': 'Nota'},
        title="Desempenho por Competência"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Detalhes por Competência
    st.markdown("<h2 class='sub-title'>Análise Detalhada por Competência</h2>", unsafe_allow_html=True)
    for comp in competencias:
        feedback_start = resultado.find(f"{comp}:\n") + len(f"{comp}:\n")
        feedback_end = resultado.find("\n\n", feedback_start)
        feedback = resultado[feedback_start:feedback_end].strip()
        with st.expander(f"{comp} - Nota: {feedback.split('\n')[0]}"):
            st.markdown(f"<div class='feedback-text'>{feedback}</div>", unsafe_allow_html=True)

    # Comentário Geral
    comentario_geral_start = resultado.find("Comentário Geral:") + len("Comentário Geral:")
    comentario_geral_end = resultado.find("Competências que precisam de mais atenção:", comentario_geral_start)
    comentario_geral = resultado[comentario_geral_start:comentario_geral_end].strip()
    st.markdown("<h2 class='sub-title'>Análise Geral</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='feedback-text'>{comentario_geral}</div>", unsafe_allow_html=True)

    # Competências que precisam de mais atenção
    atencao_start = resultado.find("Competências que precisam de mais atenção:") + len("Competências que precisam de mais atenção:")
    atencao_end = resultado.find("Redação analisada:", atencao_start)
    competencias_atencao = resultado[atencao_start:atencao_end].strip()
    st.markdown("<h2 class='sub-title'>Competências que Precisam de Mais Atenção</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='warning-box'>{competencias_atencao}</div>", unsafe_allow_html=True)

    # Botões de navegação
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Voltar para Entrada de Dados"):
            st.session_state['page'] = 'input'
            st.experimental_rerun()
    with col2:
        if st.button("Iniciar Trilha de Correção Personalizada"):
            st.session_state['page'] = 'trilha'
            st.experimental_rerun()


def display_trilha_page():
    st.markdown("<h1 class='main-title'>Trilha de Correção Personalizada</h1>", unsafe_allow_html=True)

    if 'competencia_atual' not in st.session_state:
        st.session_state['competencia_atual'] = 0

    competencias = list(st.session_state['resultado']['competencias'].keys())

    if st.session_state['competencia_atual'] < len(competencias):
        competencia = competencias[st.session_state['competencia_atual']]
        feedback_original = st.session_state['resultado']['competencias'][competencia]['feedback']
        
        nivel_proficiencia, analise_final = trilha_correcao_personalizada(
            st.session_state['qa_chain'],
            competencia,
            st.session_state['resultado']['redacao_texto'],
            feedback_original
        )

        if nivel_proficiencia and analise_final:
            st.markdown(f"<div class='success-box'>", unsafe_allow_html=True)
            st.markdown(f"<h3>Nível de Proficiência Final: {nivel_proficiencia}</h3>", unsafe_allow_html=True)
            st.markdown("### Análise Final:")
            st.write(analise_final)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state['competencia_atual'] += 1
            if st.session_state['competencia_atual'] < len(competencias):
                if st.button("Continuar para a próxima competência"):
                    st.experimental_rerun()
            else:
                st.markdown("<div class='success-box'>Parabéns! Você completou a trilha de correção para todas as competências.</div>", unsafe_allow_html=True)
                if st.button("Voltar para a Análise"):
                    st.session_state['page'] = 'analysis'
                    st.experimental_rerun()
    else:
        st.markdown("<div class='success-box'>Você completou a trilha de correção para todas as competências.</div>", unsafe_allow_html=True)
        if st.button("Voltar para a Análise"):
            st.session_state['page'] = 'analysis'
            st.experimental_rerun()

if __name__ == "__main__":
    main()
