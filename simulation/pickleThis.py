import os
import pickle


def pickleThis(fn):  # define a decorator for a function "fn"
    def wrapped(self, *args, **kwargs):  # define a wrapper that will finally call "fn" with all arguments
        # if cache exists -> load it and return its content
        cachefile = None
        if args and args[0] in self.pickles:
            pickle_name = self.pickles[args[0]]
            cachefile = self.parameters['output_path'] + 'pickles/' + pickle_name

        if cachefile and os.path.exists(cachefile):
            with open(cachefile, 'rb') as cachehandle:
                #print("using pickled result from '%s'" % cachefile)
                return pickle.load(cachehandle)

        # execute the function with all arguments passed

        res = fn(self, *args, **kwargs)

        pickle_name = str(self.pickle_count) + '.p'
        self.pickle_count += 1
        cachefile = self.parameters['output_path'] + 'pickles/' + pickle_name

        # write to cache file
        #print ("trying to write to {1}".format(cachefile))
        try:
            with open(cachefile, 'wb') as cachehandle:
                pickle.dump(res, cachehandle)
                self.pickles[args[0]] = pickle_name
        except AttributeError as e:
            print (": dumping {0} to {1} with error {2}".format(res,pickle_name,e))


        return res

    return wrapped
