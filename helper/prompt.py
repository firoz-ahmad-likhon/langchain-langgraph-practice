class Prompt:
    """Prompt class for generating prompts for different tasks."""

    @staticmethod
    def prompt_cv() -> str:
        """Prompt for CV extraction."""
        return """
            You are a human resource specialist who is responsible for reviewing candidates' CVs. You will be given the CV of the candidate and your job is to extract the information mentioned below. Also, you must follow the desired output.

            Only extract the properties mentioned in the 'CV' function or schema.

            Note: if any of the information is not mentioned in the cv, just leave it blank (empty string). Fix the word case - use camel case when appropriate. Use proper punctuation and word space. Remove the icon, extra space, line breaks, bullet lists. For example 'S O FT W A R E E N G IN E E R' would be 'Software Engineer' after removing the space and fix the case.
            """

    @staticmethod
    def prompt_classification() -> str:
        """Prompt for classification extraction."""
        return """
        Extract the desired information from the following passage.

        Only extract the properties mentioned in the 'Classification' function or schema.
        """
