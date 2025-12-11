import SwiftUI

@available(macOS 26, iOS 26, *)
struct QuestionView: View {
    @EnvironmentObject private var model: AppModel
    @State private var localQuestion: String = ""

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Step 3 - Question-driven map")
                    .font(.title2.bold())
                Text("Ask high-level questions like \"What are the main approaches to X?\". The app retrieves the best chunks across papers, then lets the on-device model synthesize a mini-survey.")
                    .foregroundStyle(.secondary)

                if model.papers.isEmpty {
                    Text("You need some ingested papers first.")
                        .foregroundStyle(.secondary)
                    Spacer()
                } else {
                    HStack(alignment: .top) {
                        TextField("Ask a question about your corpus...", text: $localQuestion, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                        Button {
                            let question = localQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !question.isEmpty else { return }
                            model.recordQuestionAsked(question)
                            model.answerQuestion(question)
                        } label: {
                            Label("Ask", systemImage: "paperplane.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isAnswering || localQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }

                    if model.isAnswering {
                        HStack {
                            ProgressView()
                            Text("Thinking...")
                        }
                    }

                    if !model.questionAnswer.isEmpty {
                        Divider().padding(.vertical, 4)
                        Text("Answer")
                            .font(.headline)
                        ScrollView {
                            Text(model.questionAnswer)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 220)
                    }

                    if !model.questionTopPapers.isEmpty {
                        Divider().padding(.vertical, 4)
                        Text("Most relevant papers")
                            .font(.headline)
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(model.questionTopPapers) { scored in
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(scored.paper.title)
                                            .font(.subheadline.bold())
                                        Text(String(format: "Similarity: %.3f", scored.score))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Text(scored.paper.summary)
                                            .font(.caption2)
                                            .lineLimit(4)
                                    }
                                    .padding(8)
                                    .background(Color.green.opacity(0.06))
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                            }
                        }
                    }

                    if !model.questionEvidence.isEmpty {
                        Divider().padding(.vertical, 4)
                        DisclosureGroup("Evidence (\(model.questionEvidence.count) chunks)") {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(model.questionEvidence.prefix(12)) { ev in
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(ev.paperTitle)
                                            .font(.subheadline.bold())
                                        Text(String(format: "Chunk score: %.3f", ev.score))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Text(ev.chunk.text)
                                            .font(.caption)
                                            .lineLimit(6)
                                    }
                                    .padding(8)
                                    .background(Color.orange.opacity(0.06))
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                            }
                        }
                    }

                    Spacer()
                }
            }
            .padding()
            .navigationTitle("Q&A")
        }
    }
}
