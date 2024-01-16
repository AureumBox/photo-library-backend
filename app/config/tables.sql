-- CreateEnum
CREATE TYPE gender AS ENUM ('MALE', 'FEMALE');

-- CreateTable
CREATE TABLE authors (
    id UUID NOT NULL,
    name TEXT,
    pseudonim TEXT,
    birth_date DATE,
    death_date DATE,
    gender gender,
    parents TEXT[],
    children TEXT[],
    siblings TEXT[],

    CONSTRAINT authors_pkey PRIMARY KEY (id)
);

-- CreateTable
CREATE TABLE relevant_activities (
    id UUID NOT NULL,
    activity TEXT NOT NULL,
    type TEXT NOT NULL,
    place TEXT NOT NULL,
    place_site TEXT NOT NULL,
    activity_start_timestamp TIMESTAMP(3) NOT NULL,
    activity_end_timestamp TIMESTAMP(3) NOT NULL,
    awards_received TEXT[],
    groups TEXT[],
    magazines TEXT[],

    CONSTRAINT relevant_activities_pkey PRIMARY KEY (id)
);

-- CreateTable
CREATE TABLE phases (
    id UUID NOT NULL,
    phase TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    trend TEXT NOT NULL,
    movement TEXT NOT NULL,
    style TEXT NOT NULL,
    description TEXT NOT NULL,
    author_id UUID NOT NULL,

    CONSTRAINT phases_pkey PRIMARY KEY (id)
);

-- CreateTable
CREATE TABLE works (
    id UUID NOT NULL,
    genre TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    meter TEXT NOT NULL,
    subject TEXT NOT NULL,
    original_language TEXT NOT NULL,
    elaboration_start_date DATE NOT NULL,
    elaboration_end_date DATE NOT NULL,
    elaboration_places TEXT[],

    CONSTRAINT works_pkey PRIMARY KEY (id)
);

-- CreateTable
CREATE TABLE works_about_works (
    referencing_work_id UUID NOT NULL,
    referenced_work_id UUID NOT NULL,

    CONSTRAINT works_about_works_pkey PRIMARY KEY (referencing_work_id,referenced_work_id)
);

-- CreateTable
CREATE TABLE multimedia (
    id UUID NOT NULL,
    work_id UUID,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    tag TEXT NOT NULL,
    source TEXT NOT NULL,
    copyright TEXT NOT NULL,
    reference TEXT NOT NULL,
    author_id UUID,
    publication_id UUID,

    CONSTRAINT multimedia_pkey PRIMARY KEY (id)
);

-- CreateTable
CREATE TABLE publications (
    id UUID NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    publication_date DATE NOT NULL,
    publication_place TEXT NOT NULL,
    edition TEXT NOT NULL,
    publisher TEXT NOT NULL,
    language TEXT NOT NULL,
    translator TEXT NOT NULL,
    work_id UUID NOT NULL,

    CONSTRAINT publications_pkey PRIMARY KEY (id)
);

-- CreateTable
CREATE TABLE author_work (
    author_id UUID NOT NULL,
    work_id UUID NOT NULL,

    CONSTRAINT author_work_pkey PRIMARY KEY (author_id,work_id)
);

-- CreateTable
CREATE TABLE phase_work (
    phase_id UUID NOT NULL,
    work_id UUID NOT NULL,

    CONSTRAINT phase_work_pkey PRIMARY KEY (phase_id,work_id)
);

-- CreateTable
CREATE TABLE relevant_activity_author (
    relevant_activity_id UUID NOT NULL,
    author_id UUID NOT NULL,

    CONSTRAINT relevant_activity_author_pkey PRIMARY KEY (relevant_activity_id,author_id)
);

-- CreateIndex
CREATE UNIQUE INDEX relevant_activities_activity_activity_start_timestamp_place_key ON relevant_activities(activity, activity_start_timestamp, place, place_site);

-- CreateIndex
CREATE UNIQUE INDEX phases_author_id_start_date_key ON phases(author_id, start_date);

-- CreateIndex
CREATE UNIQUE INDEX publications_work_id_publisher_publication_date_publication_key ON publications(work_id, publisher, publication_date, publication_place);

-- AddForeignKey
ALTER TABLE phases ADD CONSTRAINT phases_author_id_fkey FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE works_about_works ADD CONSTRAINT works_about_works_referencing_work_id_fkey FOREIGN KEY (referencing_work_id) REFERENCES works(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE works_about_works ADD CONSTRAINT works_about_works_referenced_work_id_fkey FOREIGN KEY (referenced_work_id) REFERENCES works(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE multimedia ADD CONSTRAINT multimedia_work_id_fkey FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE multimedia ADD CONSTRAINT multimedia_author_id_fkey FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE multimedia ADD CONSTRAINT multimedia_publication_id_fkey FOREIGN KEY (publication_id) REFERENCES publications(id) ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE publications ADD CONSTRAINT publications_work_id_fkey FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE author_work ADD CONSTRAINT author_work_author_id_fkey FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE author_work ADD CONSTRAINT author_work_work_id_fkey FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE phase_work ADD CONSTRAINT phase_work_phase_id_fkey FOREIGN KEY (phase_id) REFERENCES phases(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE phase_work ADD CONSTRAINT phase_work_work_id_fkey FOREIGN KEY (work_id) REFERENCES works(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE relevant_activity_author ADD CONSTRAINT relevant_activity_author_relevant_activity_id_fkey FOREIGN KEY (relevant_activity_id) REFERENCES relevant_activities(id) ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE relevant_activity_author ADD CONSTRAINT relevant_activity_author_author_id_fkey FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE RESTRICT ON UPDATE CASCADE;
