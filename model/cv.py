from typing import List, Optional
from pydantic import BaseModel, Field


class Contact(BaseModel):
    """Contact model."""

    email: Optional[str] = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")
    location: Optional[str] = Field(description="Location/address")
    linkedin: Optional[str] = Field(description="LinkedIn profile URL")
    git: Optional[str] = Field(
        description="GitHub/Bitbucket/GitLab profile URL")
    website: Optional[str] = Field(description="Personal website")


class Education(BaseModel):
    """Education model."""

    degree: Optional[str] = Field(description="Degree name")
    field_of_study: Optional[str] = Field(
        description="Field of study (Extract from degree if needed)")
    institution: Optional[str] = Field(description="Institution name")
    location: Optional[str] = Field(description="Location of Institution")
    result: Optional[str] = Field(description="Result (Grade/GPA)")
    start_date: Optional[str] = Field(
        description="Start Date in YYYY-MM-DD format. If only year and month are provided, use YYYY-MM-01. If only year is provided, use YYYY-01-01.",
    )
    end_date: Optional[str] = Field(
        description="End Date in YYYY-MM-DD format. If only year and month are provided, use YYYY-MM-[last day]. If only year is provided, use YYYY-12-31.",
    )


class Experience(BaseModel):
    """Experience model."""

    position: Optional[str] = Field(
        description="Job position (Extract only recognizable job titles)")
    company: Optional[str] = Field(description="Company name")
    location: Optional[str] = Field(description="Location of the company")
    start_date: Optional[str] = Field(
        description="Start Date in YYYY-MM-DD format. If only year and month are provided, use YYYY-MM-01. If only year is provided, use YYYY-01-01.",
    )
    end_date: Optional[str] = Field(
        description="End Date in YYYY-MM-DD format. If only year and month are provided, use YYYY-MM-[last day]. If only year is provided, use YYYY-12-31. Otherwise, it will be 'Present'.",
    )
    responsibilities: Optional[str] = Field(
        description="Key responsibilities in this role")


class CV(BaseModel):
    """CV model."""

    name: str = Field(description="The full name of the candidate")
    title: str = Field(description="Professional title or role")
    contact: Contact = Field(description="Contact details of the candidate")
    education: Optional[List[Education]] = Field(
        description="Educational background")
    experience: Optional[List[Experience]] = Field(
        description="Work experience")
    skills: Optional[List[str]] = Field(
        description="List of unique skills mentioned in the CV")
    skills_from_work_experience: Optional[List[str]] = Field(
        description="List of unique skills derived from work experience")
