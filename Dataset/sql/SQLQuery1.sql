-- 1. Create the database
CREATE DATABASE FYP;
GO

-- 2. Switch to the newly created database
USE FYP;
GO

-- 3. Create the UserData table with proper data types, constraints, and primary key
CREATE TABLE dbo.UserData (
    Id                    INT             IDENTITY(1,1) PRIMARY KEY,  -- Primary key with auto-increment
    Username              NVARCHAR(50)    NOT NULL,                    -- Added Username for user identification
    [Urban/Rural]         NVARCHAR(10)    NOT NULL,
    Gender                NVARCHAR(10)    NOT NULL,
    [Frequency of SM Use] NVARCHAR(20)    NOT NULL,
    [Education Level]     NVARCHAR(50)    NOT NULL,
    Country               NVARCHAR(50)    NOT NULL,
    [Socioeconomic Status] NVARCHAR(20)   NOT NULL,
    State                 NVARCHAR(50)    NOT NULL,
    [Peer Comparison]     INT             NOT NULL,
    [Body Image Impact]   INT             NOT NULL,
    [Sleep Quality Impact] INT            NOT NULL,
    [Self Confidence]     INT             NOT NULL,
    Cyberbullying         INT             NOT NULL,
    [Anxiety Level]       INT             NOT NULL,
    [Age Category]        NVARCHAR(20)    NOT NULL,
    [Total Interaction]   INT             NOT NULL,
    [Usage Intensity]     INT             NOT NULL,
    [Usage-Anxiety]       FLOAT           NOT NULL,
    [Most Used SM Platform] NVARCHAR(20)  NOT NULL,
    [Likes Received]      INT             NOT NULL,
    [Comments Received]   INT             NOT NULL,
    predicted_class       NVARCHAR(20)    NOT NULL,                    -- Predicted class (e.g., High, Medium, Low)
    cluster               INT             NOT NULL,                    -- Cluster ID for classification
    timestamp             DATETIME        DEFAULT GETDATE()           -- Timestamp with default as current time
);
GO

-- 4. Check if the 'FYP' database exists
SELECT name
FROM sys.databases
WHERE name = 'FYP';
GO

-- 5. Sample data retrieval to check the table
SELECT TOP 5 *
FROM FYP.dbo.UserData;
GO
