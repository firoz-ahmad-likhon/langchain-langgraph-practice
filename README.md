## Introduction
A langchain and langgraph practice project.

## Prerequisites
- python (updated) installed
- pip installed
- OLLAMA installed

## DEVELOPMENT
1. Clone the repository.
2. Create a virtual environment `python -m venv .venv`.
3. Activate the virtual environment `source .venv/bin/activate`. In Windows, use `.venv\Scripts\activate` instead.
4. Install dependencies `pip install -r requirements.txt`.
5. Copy the `.env.example` to `.env` and update the values as per your environment.
6. Start the OLLAMA server `ollama serve`.
7. Create `mkdir resource/cv` directory and copy `cp resource/software-engineer-resume-example.pdf resource/cv` into it.
   In Windows, use `copy resource/software-engineer-resume-example.pdf resource/cv` instead.
8. Run the code `python file_name.py`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
