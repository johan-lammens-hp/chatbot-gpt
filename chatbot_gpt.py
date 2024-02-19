# chatbot_gpt.py
# see https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import streamlit as st
from openai import OpenAI

st.title("ChatGPT-like clone with context window management (JML)")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    #st.session_state["openai_model"] = "gpt-3.5-turbo-0613" # 4k context, for experiments
    st.session_state["openai_model"] = "gpt-3.5-turbo-0125" # 16k context
    st.session_state["max_context_len"] = 0.90 * 16 * 1024 * 5  # 90% of estimated length in characters

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# initialize conversation turn sequence number
if "turn_index" not in st.session_state:
    st.session_state.turn_index = 0

# initialize total context length counter
if "context_length" not in st.session_state:
    st.session_state.context_length = 0

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept & process user input (main thread)
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.turn_index += 1
    st.session_state.context_length += len(prompt)
    st.session_state.messages.append({
        "index": st.session_state.turn_index, 
        "role": "user", 
        "content": prompt
        })

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(f"[{st.session_state.turn_index}|{st.session_state.context_length}/{int(st.session_state.max_context_len)}] {prompt}")

    # from a chat with gpt-3.5-turbo: (!)
    #
    # supported values for "role" in gpt-3.5-turbo are "system", "user", and "assistant"
    # The system prompt is meant to guide the model's behavior or set the context for the conversation. It can be used to provide background information, instructions, or any other information you want the AI model to be aware of.
    # Note that GPT-3 does not pay strong attention to the system message, so important instructions are often better placed in a user message.
    # GPT-3, the model I am based on, has a context window of 2048 tokens.
    # [upgraded to 4096 tokens in the meantime for gpt-3.5-turbo-0613, and 16k tokens for gpt-3.5-turbo-0125 or alias gpt-3.5-turbo starting 2024-02-16 - JML]
    # As a rough estimate, the context window of GPT-3 is approximately 12,000 to 16,000 characters.
    # Assuming an average of 250-300 words per page, and an average of 5-6 characters per word, 2048 tokens would roughly correspond to about 2-3 book pages
    #
    # potential strategies for limiting context size:
    # - delete messages from the beginning (recommended), middle (not), or end (not)
    # - have GPT itself summarize the context to less than the maximum size (AKO unlimited context)
    # - the latter applied only to some part of the context, e.g. the first half or some part
    #
    # exceeding the maximum context window size will result in an error message like this:
    # BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 4097 tokens. However, your messages resulted in 10636 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}

    # remove old items (at the beginning) from the messages list if max context length exceeded
    # remove some more than strictly needed to avoid having to do this with every turn
    if st.session_state.context_length > st.session_state.max_context_len:
        while st.session_state.context_length > 0.9 * st.session_state.max_context_len:
            # decrease total by length of the first item before removing it
            st.session_state.context_length -= len(st.session_state.messages[0]["content"])
            # remove the first item
            st.session_state.messages.pop(0)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    # add response to the messages list
    st.session_state.messages.append({
        "index": st.session_state.turn_index,    # question and answer have same turn index number
        "role": "assistant", 
        "content": response
        })

    # update the context length counter
    st.session_state.context_length += len(response)
    