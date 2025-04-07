class Prompt:
    """Prompt class for generating prompts for different tasks."""

    @staticmethod
    def prompt_sql() -> str:
        """Prompt for SQL query generation."""
        return """
        You are an AI assistant that generates SQL queries for InfluxDB.
        Convert the following natural language question into an InfluxDB SQL query.
        Return only the SQL query with no explanation.
        You must adhere the desired output format.

        Desired Output: JSON format like the following:
        ###
        {
          "name": "The generated InfluxDB sql query from provided question",
        }
        """

    @staticmethod
    def prompt_cv() -> str:
        """Prompt for CV extraction."""
        return """
            You are a human resource specialist who is responsible for reviewing candidates' CVs. You will be given the CV of the candidate and your job is to extract the information mentioned below. Also, you must follow the desired output.

            Only extract the properties mentioned in the 'CV' function or schema.

            Note: if any of the information is not mentioned in the cv, just leave it blank (empty string). Fix the word case - use camel case when appropriate. Use proper punctuation and word space. Remove the icon, extra space, line breaks, bullet lists. For example 'S O FT W A R E E N G IN E E R' would be 'Software Engineer' after removing the space and fix the case.
            """

    @staticmethod
    def prompt_cv_details() -> str:
        """Prompt for CV extraction."""
        return """
                You are a human resource specialist who is responsible for reviewing candidates' CVs. You will be given the CV of the candidate and your job is to extract the information mentioned below. Also, you must follow the desired output.

                Desired Output: JSON format like the following:
                ###
                {
                  "name": "The full name of the candidate",
                  "title": "Professional title or role from anywhere in the CV"
                  "contact": {
                    "email": "Email address",
                    "phone": "Phone number",
                    "location": "Location/address",
                    "linkedin": "LinkedIn profile URL",
                    "git": "GitHub/Bitbucket/Gitlab profile URL",
                    "website": "Personal website"
                  },
                  "education": [
                    {
                      "degree": "Degree Name",
                      "field_of_study": "Field of study (Extract from degree if needed)",
                      "institution": "Institution name",
                      "location": "Location of Institution",
                      "result": "Result (Grade/GPA)",
                      "start_date": "Start Date in the format YYYY-MM-DD. If only the year and month are provided, append the first day of that month (YYYY-MM-01). If only the year is provided, set the date to January 1st of that year (YYYY-01-01).",
                      "end_date": "End Date in the format YYYY-MM-DD. If only the year and month are provided, append the last day of that month (YYYY-MM-[last day]). If only the year is provided, set the date to December 31st of that year (YYYY-12-31)."
                    }
                  ],
                  "experience": [
                    {
                      "position": "Job position (Extract only recognizable job titles)",
                      "company": "Company name",
                      "location": "Location of the company",
                      "start_date": "Start Date in the format YYYY-MM-DD. If only the year and month are provided, append the first day of that month (YYYY-MM-01). If only the year is provided, set the date to January 1st of that year (YYYY-01-01).",
                      "end_date": "End Date in the format YYYY-MM-DD. If only the year and month are provided, append the last day of that month (YYYY-MM-[last day]). If only the year is provided, set the date to December 31st of that year (YYYY-12-31). Other wise it will be present."
                      "responsibilities": "Key Responsibilities"
                    }
                  ],
                  "skills": ["List of unique skills mentioned in the CV"],
                  "skills_from_work_experience": [
                    "List of unique skills keywords derived from work experiences"
                  ]
                }
                ###

                Note: if any of the information is not mentioned in the cv, just leave it blank (empty string). Fix the word case - use camel case when appropriate. Use proper punctuation and word space. Remove the icon, extra space, line breaks, bullet lists. For example 'S O FT W A R E E N G IN E E R' would be 'Software Engineer' after removing the space and fix the case.
                """

    @staticmethod
    def prompt_classification() -> str:
        """Prompt for classification extraction."""
        return """
        Extract the desired information from the following passage.

        Only extract the properties mentioned in the 'Classification' function or schema.
        """
