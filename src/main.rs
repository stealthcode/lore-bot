#[macro_use]
extern crate tantivy;
use tantivy::{
    collector::TopDocs,
    query::{QueryParser, TermQuery},
    schema::*,
    tokenizer::*,
    {Index, IndexReader, IndexWriter, ReloadPolicy},
};

extern crate tempdir;
use tempdir::TempDir;

extern crate chrono;
use chrono::{DateTime, Utc};

use std::{
    {env},
    sync::{Arc, Mutex},
    path::Path,
    collections::HashMap,
};

use serde::{Serialize, Deserialize};

use serenity::{
    model::{
        channel::{
            Message, 
            MessageType::Regular,
        }, 
        gateway::Ready,
    },
    prelude::*,
};


trait SearchableContent {
    fn get_id(self) -> String;
    fn get_title(self) -> String;
    fn get_content(self) -> String;
}

/// Content that may be restricted for viewing or editing by publications
trait Restrictable
{
    /// Returns a publication if one exists for the given principle
    fn get_publications_for<'a>(&self, principle :&'a str) -> Option<&Publication>;

    /// Returns true if the content is readable by the given principle
    fn is_readable_by<'a>(&self, principle: &'a str) -> bool {
        self.get_publications_for(principle)
            .map_or(false, |p| p.is_readable)
    }

    /// Returns true if the content is writable by the given principle
    fn is_writable_by<'a>(&self, principle: &'a str) -> bool {
        self.get_publications_for(principle)
            .map_or(false, |p| p.is_writable)
    }
}

/// Associated with one or more content structs to authorize a principle to read or mutate all associated content
#[derive(Clone, Serialize, Deserialize)]
struct Publication {
    /// A name to uniquely identify this publication 
    name: String,
    /// The principle being authorized
    principle: String,
    /// Denotes if this publication allows read access
    is_readable: bool,
    /// Denotes if this publication allows mutation access
    is_writable: bool,
}

impl Publication {
}

/// Lore is canonical description of an entity. It may be associated with multiple journal entries. Lore 
/// entries may be unlocked or replaced through manipulation of publications.
#[derive(Default, Serialize, Deserialize)]
struct Lore {
    /// A name to uniquely identify this Lore
    name: String,
    /// A content description
    content: String,
    /// An indexed set of publications by principle
    publications: HashMap<String, Publication>,
}

impl Lore {
}

impl SearchableContent for Lore {
    fn get_id(self) -> String {
        self.name
    }

    fn get_title(self) -> String {
        self.name
    }

    fn get_content(self) -> String {
        self.content
    }
}

impl Restrictable for Lore {
    fn get_publications_for<'a>(&self, principle :&'a str) -> Option<&Publication> {
        self.publications.get(&principle.to_string())
    }
}

struct JournalEntry {
    title: String,
    body: String,
    occurred_on: DateTime<Utc>,
    publications: HashMap<String, Publication>,
}

impl Restrictable for JournalEntry {
    fn get_publications_for<'a>(&self, principle :&'a str) -> Option<&Publication> {
        self.publications.get(&principle.to_string())
    }
}

impl JournalEntry {
    // public journal => publications [{$owner, is_writable: true, is_readable: true}, {'players', is_readable: true}]
    // private journal => publications [{$owner, is_writable: true, is_readable: true}]
    fn new<'a> (title :&'a str, body :&'a str, principles :Vec<Publication>) -> JournalEntry {
        let indexed_publications = principles.into_iter().fold(
            HashMap::new(), 
            |mut m, publication| {
                m.insert(publication.principle.clone(), publication.clone());
                m
            });
        JournalEntry {
            title: title.to_string(),
            body: body.to_string(),
            occurred_on: Utc::now(),
            publications: indexed_publications,
        }
    }
}

impl Default for JournalEntry {
    fn default() -> Self {
        JournalEntry {
            title: Default::default(),
            body: Default::default(),
            occurred_on: Utc::now(),
            publications: Default::default(),
        }
    }
}

struct SearchIndex {
    reader: IndexReader,
    index_writer: IndexWriter,
    query_parser: QueryParser,
    schema: Schema,
    id_schema: Field,
    title_schema: Field,
    content_schema: Field,
}

impl SearchIndex {
    fn search(self, search :String) -> tantivy::Result<Vec<String>> {
        let searcher = self.reader.searcher();
        let query = self.query_parser.parse_query(&search)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        let mut results = Vec::new();
        for (_score, doc_address) in top_docs {
            let retrieved_doc = searcher.doc(doc_address)?;
            results.push(self.schema.to_json(&retrieved_doc));
        }
        Ok(results)
    }

    fn add<T>(mut self, entry: T) 
        where T: SearchableContent + Copy
    {
        self.index_writer.add_document(doc!(
            self.id_schema => entry.get_id(),
            self.title_schema => entry.get_title(),
            self.content_schema => entry.get_content(),
        ));
    }

    fn replace<T>(mut self, entry: T) 
        where T: SearchableContent + Copy
    {
        let id = entry.get_id();
        let id_term = Term::from_field_text(self.id_schema, &id);
        self.index_writer.delete_term(id_term);
        self.add(entry);
    }

    fn reload(mut self) -> tantivy::Result<()> {
        self.index_writer.commit()?;
        self.reader.reload()
    }

    fn extract_doc_given_id(self, id: String) -> tantivy::Result<Option<Document>> {
        let id_term = Term::from_field_text(self.id_schema, &id);
        let searcher = self.reader.searcher();
        let term_query = TermQuery::new(id_term.clone(), IndexRecordOption::Basic);
        let top_docs = searcher.search(&term_query, &TopDocs::with_limit(1))?;

        if let Some((_score, doc_address)) = top_docs.first() {
            let doc = searcher.doc(*doc_address)?;
            Ok(Some(doc))
        } else {
            // no doc matching this ID.
            Ok(None)
        }
    }
}

fn build_index<T, P>(entries: Vec<T>) -> tantivy::Result<SearchIndex> 
    where T: SearchableContent + Copy,
    P: AsRef<Path>,
{
    let index_path = TempDir::new("tantivy_example_dir")?;
    let mut schema_builder = Schema::builder();
    
    schema_builder.add_text_field("id", STRING | STORED);

    let text_field_indexing = TextFieldIndexing::default()
        .set_tokenizer("stoppy")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_field_indexing)
        .set_stored();

    schema_builder.add_text_field("title", text_options);

    let text_field_indexing = TextFieldIndexing::default()
        .set_tokenizer("stoppy")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_field_indexing);
    schema_builder.add_text_field("content", text_options);

    let text_field_indexing = TextFieldIndexing::default()
        .set_tokenizer("stoppy")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_field_indexing);
    schema_builder.add_text_field("race", text_options);

    let text_field_indexing = TextFieldIndexing::default()
        .set_tokenizer("stoppy")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_field_indexing);
    schema_builder.add_text_field("role", text_options);

    let schema = schema_builder.build();

    let index = Index::create_in_dir(index_path, schema.clone())?;

    let tokenizer = SimpleTokenizer
        .filter(LowerCaser)
        .filter(StopWordFilter::remove(vec![
            "the".to_string(),
            "and".to_string(),
        ]));

    index.tokenizers().register("stoppy", tokenizer);

    let mut index_writer = index.writer(50_000_000)?;
    let id_schema = schema.get_field("id").unwrap();
    let title_schema = schema.get_field("title").unwrap();
    let content_schema = schema.get_field("content").unwrap();

    for entry in entries {
        index_writer.add_document(doc!(
            id_schema => entry.get_id(),
            title_schema => entry.get_title(),
            content_schema => entry.get_content(),
        ));
    }

    index_writer.commit()?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommit)
        .try_into()?;
    let query_parser = QueryParser::for_index(&index, vec![title_schema, content_schema]);
    Ok(SearchIndex {
        reader: reader,
        query_parser: query_parser,
        schema: schema,
        id_schema: id_schema,
        title_schema: title_schema,
        content_schema: content_schema,
        index_writer: index_writer,
    })
}

#[derive(Clone)]
struct BotEventHandler {
    search_indices: Arc<HashMap<String, Arc<Mutex<SearchIndex>>>>,
    repo: Arc<LoreRepo>,
}

impl EventHandler for BotEventHandler {
    // Event handlers are dispatched through a threadpool, and so multiple
    // events can be dispatched simultaneously.
    fn message(&self, ctx: Context, msg: Message) {
        if msg.kind == Regular && msg.content.starts_with("!find") {
            let principle = msg.channel_id.name(ctx.cache).unwrap_or(msg.author.name);
            let response = match self.clone().handle_search(msg.content, principle) {
                Ok(response) => format!("Found: {}", response),
                Err(err_response) => format!("Error: {}", err_response),
            };
            if let Err(why) = msg.channel_id.say(&ctx.http, response) {
                println!("Error sending message: {:?}", why);
            };
        }
    }
}

impl BotEventHandler {
    fn handle_search(self, content: String, principle: String) -> Result<String, String> {
        let mut split = content.split_whitespace();
        split.next();
        if let Some(search_term) = split.next() {
            if let Some(mutex) = self.search_indices.get(&principle).clone() {
                let lock = match mutex.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                match lock.search(search_term.to_string()) {
                    Ok(top_docs) =>  Ok(top_docs.join("\n")),
                    Err(err) => Err(format!("Could not find any results. {}", err)),
                }
            } else {
                Err("Could not accquire lock for search index".to_string())
            }
        } else {
            Err("Invalid find command. Search term must be specified".to_string())
        }
    }

    // Set a handler to be called on the `ready` event. This is called when a
    // shard is booted, and a READY payload is sent by Discord. This payload
    // contains data like the current user's guild Ids, current user data,
    // private channels, and more.
    //
    // In this case, just print what the current user's username is.
    fn ready(&self, _: Context, ready: Ready) {
        println!("Bot User '{}' is connected!", ready.user.name);
    }
}

struct LoreRepo {
    
}

impl LoreRepo {
    fn initialize() -> Result<(), String> {
        // table of lore publications with collection of lore URIs
        // table of journal publications with collection of journal IDs, URIs
        // for each principle - 
        // find publications that allow read access for all principles
        //      get set of all URIs
        //      for each URI
        //      lookup content
        //      add as SearchableContent to principle's SearchIndex
        Ok(())
    }

    fn get_publications_for_principle(p: String) -> Result<Vec<Publication>, String> {
        Ok(vec![])
    }

    fn get_lore_for_publication(p: Publication) -> Result<Vec<Lore>, String> {
        Ok(vec![])
    }

    fn get_journals_for_publication(p: Publication) -> Result<Vec<JournalEntry>, String> {
        Ok(vec![])
    }

}

fn main() {
    let token = env::var("DISCORD_TOKEN")
        .expect("Expected a token in the environment");
    let eventHandler = BotEventHandler{
        search_indices: Arc::new(HashMap::new()),
        repo: Arc::new(LoreRepo{}),
    };
    let mut client = Client::new(&token, eventHandler).expect("Err creating client");
    if let Err(why) = client.start() {
        println!("Client error: {:?}", why);
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        // assert_eq!(add(1, 2), 3);
    }
}