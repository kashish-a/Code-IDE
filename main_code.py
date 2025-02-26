import streamlit as st
import ast
import unittest
import io
import sys
import traceback

# =============================================================================
# Hugging Face Model Loader (Cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_generator():
    """
    Loads the Hugging Face text-generation pipeline using the Salesforce/codegen-350M-mono model.
    This model is optimized for code generation.
    """
    from transformers import pipeline
    # Note: Use device=0 to run on GPU (if available), or device=-1 for CPU.
    generator = pipeline(
        "text-generation",
        model="Salesforce/codegen-350M-mono",
        tokenizer="Salesforce/codegen-350M-mono",
        device=-1  # Change to 0 if you have a GPU.
    )
    return generator

# =============================================================================
# AI-Driven Helper Functions
# =============================================================================
def generate_code_from_description(desc: str) -> str:
    """
    Uses a Hugging Face text-generation model to generate code from a natural language description.
    """
    generator = load_generator()
    prompt = f"# Description: {desc}\n# Code:\n"
    # Use a reasonable max_length (e.g., 256 tokens) instead of an extremely high value.
    outputs = generator(prompt, max_length=256, num_return_sequences=1, do_sample=True, temperature=0.7)
    generated_text = outputs[0]["generated_text"]
    # Try to remove the prompt from the generated output.
    if "# Code:" in generated_text:
        code = generated_text.split("# Code:")[1]
    else:
        code = generated_text
    return code.strip()

def generate_unit_tests_for_code(code: str) -> str:
    """
    Analyzes the code to extract function definitions and auto-generates simple unittest test cases.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"# Could not generate tests due to syntax error: {e}"

    tests = "import unittest\n\n"
    found_function = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found_function = True
            func_name = node.name
            num_args = len(node.args.args)
            test_func_name = f"test_{func_name}"
            tests += f"class Test_{func_name}(unittest.TestCase):\n"
            tests += f"    def {test_func_name}(self):\n"
            # Provide dummy arguments (using simple values) if needed.
            args = ", ".join(["1"] * num_args) if num_args > 0 else ""
            tests += f"        try:\n"
            tests += f"            result = {func_name}({args})\n"
            tests += f"            self.assertIsNotNone(result)\n"
            tests += f"        except Exception as e:\n"
            tests += f"            self.fail(f'Function {func_name} raised an exception: {{e}}')\n\n"

    if not found_function:
        tests += "# No function definitions found in the provided code to test.\n"

    tests += "if __name__ == '__main__':\n"
    tests += "    unittest.main()\n"
    return tests

def analyze_code(code: str) -> str:
    """
    Performs a basic static analysis by trying to compile the code.
    """
    try:
        compile(code, '<string>', 'exec')
        return "No syntax errors detected."
    except SyntaxError as e:
        return f"Syntax Error: {e}"

def suggest_bug_fix(error_message: str, code: str) -> str:
    """
    Uses the Hugging Face model to suggest a potential fix for the provided error message.
    """
    generator = load_generator()
    prompt = (
        f"# The following code produced an error:\n"
        f"# Code:\n{code}\n"
        f"# Error: {error_message}\n"
        f"# Provide a concise suggestion to fix the error in the code:\n"
    )
    # Use a shorter max_length and lower temperature for a concise suggestion.
    outputs = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)
    suggestion = outputs[0]["generated_text"]
    if "fix" in suggestion.lower():
        # Try to remove everything before the suggestion prompt.
        suggestion_text = suggestion.split("fix")[-1]
    else:
        suggestion_text = suggestion
    return suggestion_text.strip()

def debug_code(code: str) -> str:
    """
    Executes the code in a safe namespace to catch runtime exceptions,
    and provides AI-generated suggestions for bug fixes if an error occurs.
    """
    local_env = {}
    try:
        exec(code, local_env)
        return "Code executed without runtime errors."
    except Exception as e:
        error_message = str(e)
        # Capture full traceback to aid in debugging.
        tb = traceback.format_exc()
        ai_suggestion = suggest_bug_fix(error_message, code)
        return (f"Runtime Error: {error_message}\n\n"
                f"Traceback:\n{tb}\n"
                f"AI Suggested Fix:\n{ai_suggestion}")

def run_ci(user_code: str, test_code: str) -> str:
    """
    Simulates a continuous integration (CI) process by executing both the user code
    and the auto-generated tests, then reporting the results.
    """
    local_env = {}
    # Execute user code.
    try:
        exec(user_code, local_env)
    except Exception as e:
        return f"Error in user code execution: {e}"
    # Execute test code.
    try:
        exec(test_code, local_env)
    except Exception as e:
        return f"Error in test code execution: {e}"

    # Gather unittest.TestCase subclasses from the local environment.
    test_cases = []
    for obj in local_env.values():
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            test_cases.append(obj)

    if not test_cases:
        return "No test cases found to run."

    # Build and run the test suite.
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for tc in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    output = stream.getvalue()
    summary = f"\nCI Summary: Ran {result.testsRun} tests. Failures: {len(result.failures)}. Errors: {len(result.errors)}."
    return output + summary

# =============================================================================
# Streamlit App Layout and Navigation
# =============================================================================

st.set_page_config(page_title="Intelligent IDE", layout="wide")
st.title("🤖 Intelligent IDE with AI-Driven Code Generation & Debugging")
st.markdown("""
This innovative app demonstrates an **Intelligent IDE** powered by AI that:
- **Automatically generates code** from natural language descriptions using a Hugging Face model.
- **Supports in-app code editing** with static analysis.
- **Generates unit tests** automatically based on your code.
- **Offers advanced debugging** with AI-generated bug fix suggestions.
- **Simulates a continuous integration (CI) pipeline** to run your code and tests.
""")

# Sidebar Navigation
menu_options = [
    "Home",
    "Code Generation",
    "Code Editor",
    "Test Generation",
    "Debug Code",
    "CI Simulation"
]
choice = st.sidebar.radio("Navigation", menu_options)

# =============================================================================
# Home Section
# =============================================================================
if choice == "Home":
    st.subheader("Welcome to the Intelligent IDE Demo")
    st.info("""
    This app is a proof-of-concept demonstrating how AI-driven solutions can automate
    traditionally complex tasks. Explore the sidebar options to:
    - Generate code from descriptions.
    - Edit and analyze code.
    - Auto-generate and run tests.
    - Get AI-based debugging suggestions.
    - Simulate a CI workflow.
    """)

# =============================================================================
# Code Generation Section
# =============================================================================
elif choice == "Code Generation":
    st.subheader("Automated Code Generation with AI")
    description = st.text_area("Enter a description for the desired functionality:", height=150,
                               help="Example: 'Implement a calculator function' or 'Generate a Fibonacci sequence function'")
    if st.button("Generate Code"):
        with st.spinner("Generating code using AI..."):
            generated_code = generate_code_from_description(description)
        st.success("Code generated successfully!")
        st.code(generated_code, language="python")
        # Save generated code to session_state for later use.
        st.session_state['code'] = generated_code

# =============================================================================
# Code Editor Section
# =============================================================================
elif choice == "Code Editor":
    st.subheader("Code Editor")
    if 'code' not in st.session_state:
        st.session_state['code'] = "# Write your Python code here...\n"
    code_input = st.text_area("Your Code", st.session_state['code'], height=300)
    st.session_state['code'] = code_input
    st.markdown("**Static Analysis:**")
    analysis_result = analyze_code(code_input)
    st.code(analysis_result, language="text")

# =============================================================================
# Test Generation Section
# =============================================================================
elif choice == "Test Generation":
    st.subheader("Auto-Generate Unit Tests")
    if 'code' not in st.session_state or st.session_state['code'].strip() == "":
        st.warning("Please provide some code in the Code Editor first!")
    else:
        code = st.session_state['code']
        tests = generate_unit_tests_for_code(code)
        st.success("Unit tests generated based on your code!")
        st.code(tests, language="python")
        # Save tests to session_state for later CI simulation.
        st.session_state['tests'] = tests

# =============================================================================
# Debug Code Section
# =============================================================================
elif choice == "Debug Code":
    st.subheader("Advanced Debugging with AI Suggestions")
    if 'code' not in st.session_state or st.session_state['code'].strip() == "":
        st.warning("Please provide some code in the Code Editor first!")
    else:
        code = st.session_state['code']
        if st.button("Run Debugger"):
            with st.spinner("Executing code..."):
                debug_result = debug_code(code)
            st.text_area("Debug Output", debug_result, height=250)

# =============================================================================
# CI Simulation Section
# =============================================================================
elif choice == "CI Simulation":
    st.subheader("Continuous Integration Simulation")
    if 'code' not in st.session_state or st.session_state['code'].strip() == "":
        st.warning("Please provide some code in the Code Editor first!")
    else:
        code = st.session_state['code']
        # Auto-generate tests if not already done.
        if 'tests' not in st.session_state or st.session_state['tests'].strip() == "":
            tests = generate_unit_tests_for_code(code)
            st.session_state['tests'] = tests
        else:
            tests = st.session_state['tests']
        st.markdown("**User Code:**")
        st.code(code, language="python")
        st.markdown("**Generated Tests:**")
        st.code(tests, language="python")
        if st.button("Run CI Pipeline"):
            with st.spinner("Running tests..."):
                ci_output = run_ci(code, tests)
            st.text_area("CI Output", ci_output, height=300)

# =============================================================================
# Google Colab Integration (Optional)
# =============================================================================
# If you are running this in Google Colab, you can use the code below to set up an ngrok tunnel.
# Uncomment the following lines and replace YOUR_NGROK_AUTHTOKEN_HERE with your token.
#
# import os
# os.system("pip install pyngrok")
# os.system("ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN_HERE")
# from pyngrok import ngrok
#
# # Kill any previous tunnel instances (to prevent port conflicts)
# ngrok.kill()
#
# # Start Streamlit in the background (ensure this file is saved as main_code.py)
# os.system("streamlit run main_code.py &>/dev/null &")
#
# import time
# time.sleep(3)  # Give some time for Streamlit to launch
#
# # Create the ngrok tunnel
# public_url = ngrok.connect("http://localhost:8501")
#
# print(f"Streamlit App is Live at: {public_url}")
