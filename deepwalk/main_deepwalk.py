from deepwalkalgo.setting import setting
from deepwalkalgo.graph import Graph
from deepwalkalgo.deepwalk import DeepWalk
from deepwalkalgo.evaluate import evaluate

def main():
    args = setting()
    g = Graph()
    print("Reading Deepwalk...")

    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)

   
    model = DeepWalk(graph=g, path_length=args.walk_length,
                              num_paths=args.number_walks, dim=args.representation_size,
                              workers=args.workers, window=args.window_size)
    hr20, ndcg20, hr10, ndcg10, hr5, ndcg5, map_, mrr = evaluate(model.vectors, args)
    print('hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, hr5 = %.4f, ndcg5 = %.4f, map = %.4f, mrr = %.4f' % (hr20, ndcg20, hr10, ndcg10, hr5, ndcg5, map_, mrr))

               
    model.save_embeddings(args.output)

    
if __name__ == "__main__":
    main()
