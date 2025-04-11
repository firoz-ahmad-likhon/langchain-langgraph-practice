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

    @staticmethod
    def prompt_sql_query_generation() -> str:
        """Prompt for SQL query generation."""
        return """
        Given an input question, create a syntactically correct sql query to run in InfluxDB 3 to help find the answer. The sql syntax must be compatible with InfluxDB 3. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

        Only use the following tables:
        {table_info}

        Only use the following tables and their corresponding columns:
        {column_info}

        Here are sample data for each table:
        {sample_data}

        Here are some errors you should not repeat again, if any:
        {error_list}
        """
