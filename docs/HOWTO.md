# Merge Machine Walk Through...

The Merge Machine can be performant if used properly. Take a few minutes to understand how it works, it will help you get optimal results :)

### Before we start: Definitions

*source*: The dirty data you are trying to normalise
*referential*: A reference that contains all elements you are trying to identify in the source

### Before we start: How it all works 

The matching phase uses user input to figure out the optimal way of comparing character strings (we use open source [dedupe](https://github.com/datamade/dedupe)) to best fit the user input. Since we only compare character strings and MOSTLY do NOT enrich the data using external databases, our tool can only suggest matches if there are patterns that can be identified in the text. To help dedupe find these patterns, we use several steps of pre-processing to ensure that the formating of the source and of the referential are as similar as possible.

### Before we start: What we can/cannot achieve

What it will (sould) do:
- Propose matches for rows in your source file in a given referential with a reasonable precision
- Much more than just exact match
- Take much less long than what it would if done manually

What it won't do:
- Identify a match if you can't (why? if there is not enough information for you to be able to confirm that two rows are in fact the same thing, the computer won't be able to do that for you)
- Match on synonyms that are not alike (why? in the end we compare characters; if there is no similarity in the written form, it just won't work)
- Match on translations that are not alike (why? same thing)
- Match on non-equal numerical values or coordinates (why? for now, we treat numerical values like character strings, so 20000 and 19999 will be considered very different)

