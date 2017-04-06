# Merge Machine Walk Through...

The Merge Machine can be performant if used properly. Take a few minutes to understand how it works, it will help you get optimal results :-)

### Before we start: Definitions

- **source**: The dirty data you are trying to normalise
- **referential**: A reference that contains all elements you are trying to identify in the source
- **internal referential**: A frequently used referential that we pre-loaded on our server (so you don't have to do it)

### Before we start: How it all works 

The matching phase uses user input to figure out the optimal way of comparing character strings (we use open source [dedupe](https://github.com/datamade/dedupe)) to best fit the user input. Since we only compare character strings and MOSTLY do NOT enrich the data using external databases, our tool can only suggest matches if there are patterns that can be identified in the text. To help dedupe find these patterns, we use several steps of pre-processing to ensure that the formating of the source and of the referential are as similar as possible.

### Before we start: What we can/cannot achieve

What it will (should) do:
- Propose matches for rows in your source file in a given referential with a reasonable precision
- Much more than just exact match
- Take much less long than what it would if done manually

What it won't do:
- Identify a match if you can't (why? if there is not enough information for you to be able to confirm that two rows are in fact the same thing, the computer won't be able to do that for you)
- Match on synonyms that are not alike (why? in the end we compare characters; if there is no similarity in the written form, it just won't work)
- Match on translations that are not alike (why? same thing)
- Match on non-equal numerical values or coordinates (why? for now, we treat numerical values like character strings, so 20000 and 19999 will be considered very different)

## STEP 1: CREATE/LOAD YOUR PROJECT

A project contains the information for: **Matching a format of source with a format of referential**. Projects are identified by their project ID (save yours somewhere...).

When to load an existing project:
- You are uploading a new source/referential that has the same fields and the same input type as what you already uploaded
- You selected the wrong source/referential last time, you don't care about keeping the changes you made

When to create a new project:
- You want to try matching to another source/referential and don't wan't to overwrite your configuration
- You want to use a new source/referential that has a different structure (different column names / order)
- You want to use a new source/referential for which the data looks very different (addresses are not formated the same, using acronyms in stead of full names...)

## STEP 2: SELECT YOUR FILES

Select the source and the referential you will try to match to.

Choices:
- previous: Use a file you previously upload
- internal: Use a file we uploaded for you
- upload: (please be responsible) Upload a new file. Uploading a file with the same file name within a project will erase the previously uploaded file

## STEP 3: DETECT/REPLACE MISSING VALUES (OPTIONAL)

### Why?
Some people write "Missing Value" or "N/A" or "XXX" in their data to represent missing values. This can be confusing for dedupe which treats this like any other value and tries to find matches if they are not explicitly flagged as missing values.

### What should I do?
We suggest possible representations of missing values for each column and for the entire file (ALL). You should check that these values are indeed representing missing values and you can add your own if you know better (if in your data, missing values are represented by "nan" and we didn't catch that, just add "nan").

Choices:
- columns: representations of missing values only for a given column
- all: representations of missing values valid for the entire file

## STEP 4: SELECT COLUMN PAIRS

Select column matches between source and referential (addr_source <-> ADDRESS_MATCH; nomen <-> FULL_NAME, etc...)

Guidelines:
- Include columns which would suffice for a non expert human discern if a (source entity, referential entity) pair is a match
- MULTIPLE COLUMNS NOT YET IMPLEMENTED

## STEP 5: DEDUPE LABELLING

This step can be a bit long to load. Please be patient...

### Why?
[Dedupe](https://github.com/datamade/dedupe) uses ground truth examples to learn what is the best way to compare values. For example, if most of your examples are:

- 50 cent <-> 50 cents : a good metric might be the number of letter insertions between source and referential
- FCC <->  Federal Communications Commission : a good metric will compare acronyms on the right with values on the left

### What should I do?
We suggest pairs of rows in (source, referential). You should tell us whether or not these values represent the same entity.; You can choose "uncertain" when you are not sure and can use "previous" (once) if you made a mistake. Once you are done, hit next...

### What is going on?
Each time you label a pair, dedupe updates its rules.. It then proposes a new match that will best reduce uncertainty on what are good rules to use if you answer.

Guidelines:
- The more you label, the better our tool will work. Take the time, it's worth it!
- Try to be as accurate as you can. Use "uncertain" if you have a doubt. 
- Try not to use your field knowledge to label pairs. Remember: all information should somehow be included in the text...
- You might be asked to label obvious matches or non-matches. Don't mind that! Everything is going along the plan...

## STEP 6: GET RESULTS

Download the result 
select columns to match
user feedback 
