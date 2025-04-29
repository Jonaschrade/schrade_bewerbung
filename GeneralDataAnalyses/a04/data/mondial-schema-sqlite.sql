-- import to SQLite by running: sqlite3.exe db.sqlite3 -init sqlite.sql

BEGIN TRANSACTION;

-- Table for countries
CREATE TABLE IF NOT EXISTS country (
    Name TEXT NOT NULL UNIQUE,
    Code TEXT NOT NULL,
    Capital TEXT,
    Province TEXT,
    Area FLOAT,
    Population INT,
    CONSTRAINT CountryKey PRIMARY KEY(Code),
    CONSTRAINT CountryArea CHECK (Area >= 0),
    CONSTRAINT CountryPop CHECK (Population >= 0)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for cities
CREATE TABLE IF NOT EXISTS city (
    Name TEXT,
    Country TEXT,
    Province TEXT,
    Population INT,
    Longitude FLOAT,
    Latitude FLOAT,
    CONSTRAINT CityKey PRIMARY KEY (Name, Country, Province),
    CONSTRAINT CityPop CHECK (Population >= 0),
    CONSTRAINT CityLon CHECK ((Longitude >= -180) AND (Longitude <= 180)),
    CONSTRAINT CityLat CHECK ((Latitude >= -90) AND (Latitude <= 90))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for provinces
CREATE TABLE IF NOT EXISTS province (
    Name TEXT NOT NULL,
    Country TEXT NOT NULL,
    Population INT,
    Area FLOAT,
    Capital TEXT,
    CapProv TEXT,
    CONSTRAINT PrKey PRIMARY KEY (Name, Country),
    CONSTRAINT PrPop CHECK (Population >= 0),
    CONSTRAINT PrAr CHECK (Area >= 0)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for economy data
CREATE TABLE IF NOT EXISTS economy (
    Country TEXT,
    GDP FLOAT,
    Agriculture FLOAT,
    Service FLOAT,
    Industry FLOAT,
    Inflation FLOAT,
    CONSTRAINT EconomyKey PRIMARY KEY(Country),
    CONSTRAINT EconomyGDP CHECK (GDP >= 0)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for population data
CREATE TABLE IF NOT EXISTS population (
    Country TEXT,
    Population_Growth FLOAT,
    Infant_Mortality FLOAT,
    CONSTRAINT PopKey PRIMARY KEY(Country)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for political data
CREATE TABLE IF NOT EXISTS politics (
    Country TEXT,
    Independence DATE,
    Dependent TEXT,
    Government TEXT,
    CONSTRAINT PoliticsKey PRIMARY KEY(Country)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for languages
CREATE TABLE IF NOT EXISTS language (
    Country TEXT,
    Name TEXT,
    Percentage FLOAT,
    CONSTRAINT LanguageKey PRIMARY KEY (Name, Country),
    CONSTRAINT LanguagePercent CHECK ((Percentage > 0) AND (Percentage <= 100))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for religions
CREATE TABLE IF NOT EXISTS religion (
    Country TEXT,
    Name TEXT,
    Percentage FLOAT,
    CONSTRAINT ReligionKey PRIMARY KEY (Name, Country),
    CONSTRAINT ReligionPercent CHECK ((Percentage > 0) AND (Percentage <= 100))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for ethnic groups
CREATE TABLE IF NOT EXISTS ethnicGroup (
    Country TEXT,
    Name TEXT,
    Percentage FLOAT,
    CONSTRAINT EthnicKey PRIMARY KEY (Name, Country),
    CONSTRAINT EthnicPercent CHECK ((Percentage > 0) AND (Percentage <= 100))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for continents
CREATE TABLE IF NOT EXISTS continent (
    Name TEXT,
    Area FLOAT(10),
    CONSTRAINT ContinentKey PRIMARY KEY(Name)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for borders
CREATE TABLE IF NOT EXISTS borders (
    Country1 TEXT,
    Country2 TEXT,
    Length FLOAT,
    CONSTRAINT LengthCheck CHECK (Length > 0),
    CONSTRAINT BorderKey PRIMARY KEY (Country1, Country2)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for countries encompassing continents
CREATE TABLE IF NOT EXISTS encompasses (
    Country TEXT NOT NULL,
    Continent TEXT NOT NULL,
    Percentage FLOAT,
    CONSTRAINT PercentageCheck CHECK ((Percentage > 0) AND (Percentage <= 100)),
    CONSTRAINT EncompassesKey PRIMARY KEY (Country, Continent)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for organizations
CREATE TABLE IF NOT EXISTS organization (
    Abbreviation TEXT PRIMARY KEY,
    Name TEXT NOT NULL,
    City TEXT,
    Country TEXT,
    Province TEXT,
    Established DATE,
    CONSTRAINT OrgNameUnique UNIQUE (Name)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for organization memberships
CREATE TABLE IF NOT EXISTS isMember (
    Country TEXT,
    Organization TEXT,
    Type TEXT DEFAULT 'member',
    CONSTRAINT MemberKey PRIMARY KEY (Country, Organization)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for mountains
CREATE TABLE IF NOT EXISTS mountain (
    Name TEXT,
    Mountains TEXT,
    Height FLOAT,
    Type TEXT,
    Longitude FLOAT,
    Latitude FLOAT,
    CONSTRAINT MountainKey PRIMARY KEY(Name),
    CONSTRAINT LongLat_Check CHECK ((Longitude >= -180) AND (Longitude <= 180) AND (Latitude >= -90) AND (Latitude <= 90))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for deserts
CREATE TABLE IF NOT EXISTS desert (
    Name TEXT,
    Area FLOAT,
    Longitude FLOAT,
    Latitude FLOAT,
    CONSTRAINT DesertKey PRIMARY KEY(Name),
    CONSTRAINT DesCoord CHECK ((Longitude >= -180) AND (Longitude <= 180) AND (Latitude >= -90) AND (Latitude <= 90))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for islands
CREATE TABLE IF NOT EXISTS island (
    Name TEXT,
    Islands TEXT,
    Area FLOAT,
    Height FLOAT,
    Type TEXT,
    Longitude FLOAT,
    Latitude FLOAT,
    CONSTRAINT IslandKey PRIMARY KEY(Name),
    CONSTRAINT IslandAr CHECK (Area >= 0),
    CONSTRAINT IslandCoord CHECK ((Longitude >= -180) AND (Longitude <= 180) AND (Latitude >= -90) AND (Latitude <= 90))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for lakes
CREATE TABLE IF NOT EXISTS lake (
    Name TEXT,
    Area FLOAT,
    Depth FLOAT,
    Altitude FLOAT,
    Type TEXT,
    River TEXT,
    Longitude FLOAT,
    Latitude FLOAT,
    CONSTRAINT LakeKey PRIMARY KEY(Name),
    CONSTRAINT LakeAr CHECK (Area >= 0),
    CONSTRAINT LakeDpth CHECK (Depth >= 0),
    CONSTRAINT LakeCoord CHECK ((Longitude >= -180) AND (Longitude <= 180) AND (Latitude >= -90) AND (Latitude <= 90))
);

COMMIT;

BEGIN TRANSACTION;

-- Table for seas
CREATE TABLE IF NOT EXISTS sea (
    Name TEXT,
    Depth FLOAT,
    CONSTRAINT SeaKey PRIMARY KEY(Name),
    CONSTRAINT SeaDepth CHECK (Depth >= 0)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for rivers
CREATE TABLE IF NOT EXISTS river (
    Name TEXT,
    River TEXT,
    Lake TEXT,
    Sea TEXT,
    Length FLOAT,
    SourceLongitude FLOAT,
    SourceLatitude FLOAT,
    Mountains TEXT,
    SourceAltitude FLOAT,
    EstuaryLongitude FLOAT,
    EstuaryLatitude FLOAT,
    CONSTRAINT RiverKey PRIMARY KEY(Name),
    CONSTRAINT RiverLength CHECK (Length >= 0),
    CONSTRAINT SourceCoord CHECK ((SourceLongitude >= -180) AND (SourceLongitude <= 180) AND (SourceLatitude >= -90) AND (SourceLatitude <= 90)),
    CONSTRAINT EstCoord CHECK ((EstuaryLongitude >= -180) AND (EstuaryLongitude <= 180) AND (EstuaryLatitude >= -90) AND (EstuaryLatitude <= 90))
);

COMMIT;

BEGIN TRANSACTION;

-- Geo-tables for mountains, deserts, islands, rivers, seas, lakes, and estuaries
CREATE TABLE IF NOT EXISTS geo_mountain (
    Mountain TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GMountainKey PRIMARY KEY (Province, Country, Mountain)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_desert (
    Desert TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GDesertKey PRIMARY KEY (Province, Country, Desert)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_island (
    Island TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GIslandKey PRIMARY KEY (Province, Country, Island)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_river (
    River TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GRiverKey PRIMARY KEY (Province, Country, River)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_sea (
    Sea TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GSeaKey PRIMARY KEY (Province, Country, Sea)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_lake (
    Lake TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GLakeKey PRIMARY KEY (Province, Country, Lake)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_source (
    River TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GSourceKey PRIMARY KEY (Province, Country, River)
);

COMMIT;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS geo_estuary (
    River TEXT,
    Country TEXT,
    Province TEXT,
    CONSTRAINT GEstuaryKey PRIMARY KEY (Province, Country, River)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for merging seas
CREATE TABLE IF NOT EXISTS mergesWith (
    Sea1 TEXT,
    Sea2 TEXT,
    CONSTRAINT MergesWithKey PRIMARY KEY (Sea1, Sea2)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for cities located at specific geographic features
CREATE TABLE IF NOT EXISTS located (
    City TEXT,
    Province TEXT,
    Country TEXT,
    River TEXT,
    Lake TEXT,
    Sea TEXT
);

COMMIT;

BEGIN TRANSACTION;

-- Table for cities located on islands
CREATE TABLE IF NOT EXISTS locatedOn (
    City TEXT,
    Province TEXT,
    Country TEXT,
    Island TEXT,
    CONSTRAINT locatedOnKey PRIMARY KEY (City, Province, Country, Island)
);

COMMIT;

BEGIN TRANSACTION;

-- Table for islands in seas, lakes, or rivers
CREATE TABLE IF NOT EXISTS islandIn (
    Island TEXT,
    Sea TEXT,
    Lake TEXT,
    River TEXT
);

COMMIT;

BEGIN TRANSACTION;

-- Table for mountains located on islands
CREATE TABLE IF NOT EXISTS mountainOnIsland (
    Mountain TEXT,
    Island TEXT,
    CONSTRAINT MntIslKey PRIMARY KEY (Mountain, Island)
);

COMMIT;


